from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn.functional as F
import gc
from .utils import apply_rotary_pos_emb, layer_norm, topp_temperature_decode
import flashinfer
from .attnserver import LSHSparseAttnServer, AttnServer
import time
from ..quantization.awq_utils import AwqLinear
from abc import ABC, abstractmethod

class BaseLLMLayer(ABC):
    def __init__(self, layer_idx: int) -> None:
        # Common parameters
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.cos_cache: torch.Tensor = None
        self.sin_cache: torch.Tensor = None

        self.layer_idx = layer_idx

    @abstractmethod
    def init_parameters(self, hf_layer):
        """Abstract method to initialize parameters from a given layer."""
        pass

    @abstractmethod
    def init_gpu(self, device: str = 'cuda:0'):
        """Abstract method to move parameters to GPU."""
        pass

class LLMLayer(BaseLLMLayer):
    def init_parameters(self, hf_layer):
        # Initialize parameters with type hints
        self.wq: torch.Tensor = hf_layer.self_attn.q_proj.weight.detach()
        self.wk: torch.Tensor = hf_layer.self_attn.k_proj.weight.detach()
        self.wv: torch.Tensor = hf_layer.self_attn.v_proj.weight.detach()
        self.wo: torch.Tensor = hf_layer.self_attn.o_proj.weight.detach()

        self.gate_proj: torch.Tensor = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj: torch.Tensor = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj: torch.Tensor = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight: torch.Tensor = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight: torch.Tensor = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def init_gpu(self, device: str = 'cuda:0'):
        # Move parameters to the specified GPU device
        self.input_layernorm_weight: torch.Tensor = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight: torch.Tensor = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq: torch.Tensor = self.wq.to(device, non_blocking=True)
        self.wk: torch.Tensor = self.wk.to(device, non_blocking=True)
        self.wv: torch.Tensor = self.wv.to(device, non_blocking=True)
        self.wo: torch.Tensor = self.wo.to(device, non_blocking=True)
        self.gate_proj: torch.Tensor = self.gate_proj.to(device, non_blocking=True)
        self.up_proj: torch.Tensor = self.up_proj.to(device, non_blocking=True)
        self.down_proj: torch.Tensor = self.down_proj.to(device, non_blocking=True)

class LLMAwqLayer(BaseLLMLayer):
    def __init__(self, layer_idx: int) -> None:
        super().__init__(layer_idx)
        self.wq = AwqLinear()
        self.wk = AwqLinear()
        self.wv = AwqLinear()
        self.wo = AwqLinear()

        self.gate_proj = AwqLinear()
        self.up_proj = AwqLinear()
        self.down_proj = AwqLinear()

    def init_parameters(self, hf_layer):
        self.wq.init_parameters(hf_layer.self_attn.q_proj)
        self.wk.init_parameters(hf_layer.self_attn.k_proj)
        self.wv.init_parameters(hf_layer.self_attn.v_proj)
        self.wo.init_parameters(hf_layer.self_attn.o_proj)
        self.gate_proj.init_parameters(hf_layer.mlp.gate_proj)
        self.up_proj.init_parameters(hf_layer.mlp.up_proj)
        self.down_proj.init_parameters(hf_layer.mlp.down_proj)

        self.input_layernorm_weight: torch.Tensor = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight: torch.Tensor = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def init_gpu(self, device: str = 'cuda:0'):
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj = self.down_proj.to(device, non_blocking=True)



class LLM:
    def __init__(self, 
        model_name: str,
        K: int = 10,
        L: int = 150,
        batch_size :int = 1,
        max_length :int = 256, 
        generation_buffer: int = 256,
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
    
        self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.eos_tokens = self.config.eos_token_id if (isinstance(self.config.eos_token_id, list)) else [self.config.eos_token_id]
        if K > 0:
            self.attention_server = LSHSparseAttnServer(config=self.config, K=K, L=L, batch_size=self.batch_size, 
        max_length=self.max_length, device=self.device, dtype=self.dtype, generation_buffer=generation_buffer)
        else:
            self.attention_server = AttnServer(config=self.config, K=K, L=L, batch_size=self.batch_size, 
        max_length=self.max_length, device=self.device, dtype=self.dtype, generation_buffer=generation_buffer)
            
        self.k_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=self.dtype, device=self.device)
        self.chunk_size = 8192
        self.wrt_stream = torch.cuda.Stream()
    
    def create_layer(self, idx):
        return LLMLayer(idx)

    def init_parameters(self):

        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        self.layers :list[BaseLLMLayer] = []
        
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = self.create_layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)
    
    def apply_linear_layer(self, input, w):
        assert isinstance(input, torch.Tensor), f"Expected input to be torch.Tensor, got {type(input)}"
        assert isinstance(w, torch.Tensor), f"Expected w to be torch.Tensor, got {type(w)}"
        return F.linear(input, w)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        layer: BaseLLMLayer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.apply_linear_layer(hidden_states, layer.wq)
        key_states = self.apply_linear_layer(hidden_states, layer.wk)
        value_states = self.apply_linear_layer(hidden_states, layer.wv)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        layer: BaseLLMLayer
    ):  
        hidden_states = F.linear(attn_output, layer.wo)    
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        up = self.apply_linear_layer(hidden_states, layer.up_proj)
        gate = self.apply_linear_layer(hidden_states, layer.gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = self.apply_linear_layer(hidden_states, layer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: BaseLLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):
        
        residual = hidden_states
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        
        key_states = apply_rotary_pos_emb(key_states, self.cos_cache, self.sin_cache, position_ids)
        query_states = apply_rotary_pos_emb(query_states, self.cos_cache, self.sin_cache, position_ids)
        
        hidden_states = self.attention_server.decode(query_states, key_states, value_states, layer_idx)
        
        hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    @torch.inference_mode()
    def layer_prefill(self, 
            buffer: BaseLLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor,
            request_id: int = 0):

        with torch.cuda.stream(self.wrt_stream):
            residual = hidden_states
            
            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(hidden_states[:,start:end,:], buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
                bsz, q_len, _ = h.size()
                query_states = self.apply_linear_layer(h, buffer.wq)
                key_states = self.apply_linear_layer(h, buffer.wk)
                value_states = self.apply_linear_layer(h, buffer.wv)
                query_states = query_states.view(q_len, self.num_heads, self.head_dim)
                key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
                value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)
                
                key_states = apply_rotary_pos_emb(key_states, self.cos_cache, self.sin_cache, position_ids[start:end])
                query_states = apply_rotary_pos_emb(query_states, self.cos_cache, self.sin_cache, position_ids[start:end])
                
                self.k_cache[start:end].copy_(key_states)
                self.v_cache[start:end].copy_(value_states)
                
                
                h = flashinfer.prefill.single_prefill_with_kv_cache(
                    q=query_states,
                    k=self.k_cache[:end],
                    v=self.v_cache[:end],
                    causal=True,
                    allow_fp16_qk_reduction=True,
                    kv_layout="NHD"
                )
                
                h = h.reshape(bsz, q_len, self.hidden_size)
                h = self.apply_linear_layer(h, buffer.wo)
                residual[:,start:end,:].add_(h)
        
        if layer_idx >= 1:
            self.attention_server.build_table(layer_idx - 1, request_id, self.chunk_end[-1])
        
        self.wrt_stream.synchronize()
        
        
        
        
        with torch.cuda.stream(self.wrt_stream):
            hidden_states = residual
            for (start, end) in zip(self.chunk_start, self.chunk_end):
                h = layer_norm(hidden_states[:,start:end,:], buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
                up = self.apply_linear_layer(h, buffer.up_proj)
                gate = self.apply_linear_layer(h, buffer.gate_proj)
                gate = F.silu(gate)
                h = gate * up
                h = self.apply_linear_layer(h, buffer.down_proj)
                residual[:,start:end,:].add_(h)

        self.attention_server.fill(layer_idx, request_id, self.k_cache, self.v_cache, self.chunk_end[-1])
        if layer_idx == self.num_layers - 1:
            self.attention_server.build_table(layer_idx, request_id, self.chunk_end[-1])
        self.wrt_stream.synchronize()
        return residual
    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):
        
        self.attention_server.plan()
        hidden_states = F.embedding(input_ids, self.embed_tokens)
       
        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)

        hidden_states = layer_norm(hidden_states[:,-1:,:], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits
    
    @torch.inference_mode()
    def prefill(self,
        input_ids: torch.LongTensor,
        request_id : int = 0):
          

        hidden_states = F.embedding(input_ids, self.embed_tokens)
        
        self.num_chunk = ((input_ids.shape[1] // self.chunk_size ) if (input_ids.shape[1] % self.chunk_size  > 0) else (input_ids.shape[1] // self.chunk_size  - 1)) + 1
        self.chunk_start = [i * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end = [(i+1) * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end[-1] = input_ids.shape[1]
        self.attention_server.alloc_buffer(input_ids.shape[1])
        
        
        position_ids = torch.arange(input_ids.shape[1], device=self.device, dtype=torch.int32)
        for idx in range(self.num_layers):
                torch.cuda.synchronize()
                hidden_states = self.layer_prefill(self.layers[idx], idx, hidden_states, position_ids, request_id=request_id)

        hidden_states = layer_norm(hidden_states[:,-1:,:], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def generate(self,
        input_ids: torch.LongTensor, 
        max_tokens: int = 128,
        temperature: float = 0.6,
        topp: float = 0.9,
        verbose: bool = False):
        
        generated = []
        prefix_len = input_ids.shape[1]
        position_ids = torch.arange(prefix_len + max_tokens, device=self.device).unsqueeze(0)
        logits = self.prefill(input_ids=input_ids)
        torch.cuda.synchronize()
        if verbose:
            t1 = time.time()
        for k in range(max_tokens):
            if temperature < 0.1:
                input_ids = logits.argmax(dim=-1)
            else:
                input_ids = topp_temperature_decode(logits, temperature, topp)
            logits = self.inference(input_ids=input_ids, position_ids=position_ids[:,prefix_len + k:prefix_len + k + 1])
            generated.append(input_ids[0].item())
            if input_ids[0].item() in self.eos_tokens:
                break
        if verbose:
            torch.cuda.synchronize()
            t2 = time.time()
            print("\033[94m[INFO] Prefill {} tokens\033[0m".format(prefix_len))
            print("\033[94m[INFO] Generate {} tokens\033[0m".format(len(generated)))
            print("\033[94m[INFO] Decoding Latency {:.2f} ms/token\033[0m".format(1000 * (t2 - t1)/len(generated)))
        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()
        return generated
    
    def clear(self):
        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()


## This class is used for activation aware quantisation
class LLMAwq(LLM):
    def create_layer(self, idx):
        return LLMAwqLayer(idx)
    
    def apply_linear_layer(self, input, w):
        assert isinstance(input, torch.Tensor), f"Expected input to be torch.Tensor, got {type(input)}"
        assert isinstance(w, AwqLinear), f"Expected w to be torch.Tensor or AwqLinear, got {type(w)}"
        return w.apply(input)
