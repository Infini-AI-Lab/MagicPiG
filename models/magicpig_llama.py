from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from time import sleep
import math
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union
import gc
from torch.nn.attention import SDPBackend, sdpa_kernel
from utils import apply_rotary_pos_emb, layer_norm, repeat_kv
from cache import KV_Cache
from tqdm import tqdm
class LLMLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.cos_cache :torch.Tensor = None
        self.sin_cache :torch.Tensor = None

        self.layer_idx = layer_idx
    
    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)



class LLM:
    def __init__(self, 
        model_name: str,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16,
        magicpig_config = None) -> None:
        self.magicpig_config = magicpig_config
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
        self.window_size = self.magicpig_config["window_size"]
        self.mem_efficient = True if (self.magicpig_config["mem_efficient"] == 1) else False
        self.kv_cache = KV_Cache(self.config, max_length=max_length, device=device, dtype=dtype, batch_size=self.batch_size,
                                cache_layers=self.magicpig_config["cache_layers"], K=self.magicpig_config["lsh_K"], L=self.magicpig_config["lsh_L"],
                                mem_efficient=self.mem_efficient, device_budget=self.magicpig_config["device_budget"], generate_buffer=self.magicpig_config["generation_budget"])
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

        self.layers :list[LLMLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LLMLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)
        
        
    
    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        input_layernorm_variance_epsilon: float,
        input_layernorm_weight: torch.Tensor,
        wq:torch.Tensor,
        wk:torch.Tensor,
        wv:torch.Tensor,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, input_layernorm_variance_epsilon, input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, wq)
        key_states = F.linear(hidden_states, wk)
        value_states = F.linear(hidden_states, wv)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        post_attention_layernorm_variance_epsilon: float,
        post_attention_layernorm_weight: torch.Tensor,
        wo: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):  
    
    
        hidden_states = F.linear(attn_output, wo) 
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, post_attention_layernorm_variance_epsilon, post_attention_layernorm_weight)
        up = F.linear(hidden_states, up_proj)
        gate = F.linear(hidden_states, gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, down_proj)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: LLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer.input_layernorm_variance_epsilon,
            buffer.input_layernorm_weight,
            buffer.wq,
            buffer.wk,
            buffer.wv,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids)
        
        hidden_states = self.kv_cache.attention_compute(query_states, key_states, value_states, layer_idx)

        hidden_states = self.post_attention_compute(
                        hidden_states, residual,
                        buffer.post_attention_layernorm_variance_epsilon,
                        buffer.post_attention_layernorm_weight,
                        buffer.wo,
                        buffer.gate_proj,
                        buffer.up_proj,
                        buffer.down_proj,
                        )
        
        return hidden_states
    
    @torch.inference_mode()
    def layer_prefill(self, 
            buffer: LLMLayer,
            layer_idx :int, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor):

        for (start, end) in zip(self.chunk_start, self.chunk_end):

            hidden_states = self.device_h_cache[:,start:end,:]
            pos = position_ids[:,start:end]

            bsz, q_len, _ = hidden_states.size()
            query_states, key_states, value_states = self.pre_attention_compute(
                hidden_states,
                buffer.input_layernorm_variance_epsilon,
                buffer.input_layernorm_weight,
                buffer.wq,
                buffer.wk,
                buffer.wv,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim
            )
            
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, pos)
            
            key_states, value_states = self.kv_cache.prefill(key_states, value_states, start, end)
            if self.mem_efficient:
                
                query_states = query_states.reshape(self.batch_size, self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
                mask = attention_mask[..., -q_len:,-key_states.shape[-2]:].repeat(1, 1, self.num_key_value_groups, 1)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    hidden_states = F.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=mask
                    )
                hidden_states = hidden_states.reshape(bsz, self.num_heads, q_len, -1)
                if end == self.chunk_end[-1]:
                    if layer_idx not in self.kv_cache.cache_layers:
                        self.kv_cache.offload_kv_cache(layer_idx, end, window_size=self.window_size)
                    else:
                        self.kv_cache.offload_kv_cache(layer_idx, end)
                
            else:
                mask = attention_mask[..., -q_len:,-key_states.shape[-2]:]
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = attn_weights + mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                hidden_states = torch.matmul(attn_weights, value_states)
                
                if end == self.chunk_end[-1]:
                    if layer_idx not in self.kv_cache.cache_layers:
                        self.kv_cache.offload_kv_cache(layer_idx, end, window_size=self.window_size)       
                    else:
                        self.kv_cache.offload_kv_cache(layer_idx, end)
            
                

            
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
            
            hidden_states = F.linear(hidden_states, buffer.wo)
                
            self.device_h_cache[:,start:end,:] += hidden_states
            hidden_states = self.device_h_cache[:,start:end,:]
            hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
            up = F.linear(hidden_states, buffer.up_proj)
            gate = F.linear(hidden_states, buffer.gate_proj)
            F.silu(gate, inplace=True)
            hidden_states = gate * up
            hidden_states = F.linear(hidden_states, buffer.down_proj)
            self.device_h_cache[:,start:end,:] += hidden_states

        
        


    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_variance_epsilon)
        hidden_states = self.norm_weight * hidden_states.to(input_dtype)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def prefill(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):
        
        
        self.chunk_size = self.magicpig_config["chunk_size"]
        self.num_chunk = ((input_ids.shape[1] // self.chunk_size ) if (input_ids.shape[1] % self.chunk_size  > 0) else (input_ids.shape[1] // self.chunk_size  - 1)) + 1

        self.chunk_start = [i * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end = [(i+1) * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end[-1] = input_ids.shape[1]
        if self.mem_efficient:
            attention_mask = torch.ones((self.chunk_size, input_ids.shape[1]), dtype=torch.bool, device=self.device)
            attention_mask.tril_(diagonal=input_ids.shape[1] - self.chunk_size)
            attention_mask = attention_mask[None, None, :, :]
        else:
            attention_mask = torch.zeros((self.chunk_size, input_ids.shape[1]), dtype=self.dtype, device=self.device)
            byte_mask = torch.ones((self.chunk_size, input_ids.shape[1]), dtype=torch.bool, device=self.device)
            byte_mask.tril_(diagonal=input_ids.shape[1] - self.chunk_size)
            attention_mask = attention_mask.masked_fill_(~byte_mask, torch.finfo(self.dtype).min)
            attention_mask = attention_mask[None, None, :, :]
        self.device_h_cache = F.embedding(input_ids, self.embed_tokens)
        for idx in tqdm(range(self.num_layers)):
                self.layers[idx].init_gpu(self.device)
                self.layer_prefill(self.layers[idx], idx, position_ids, attention_mask)
                self.layers[idx].init_gpu("cpu")
        hidden_states = self.device_h_cache[:,-1:,:]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_variance_epsilon)
        hidden_states = self.norm_weight * hidden_states.to(input_dtype)
        logits = F.linear(hidden_states, self.lm_head).float()

        self.device_h_cache = None
        self.kv_cache.prepare_for_decode()
        for idx in range(self.num_layers):
                self.layers[idx].init_gpu(self.device)
        return logits

class LLMEngine:
    def __init__(self, 
                model_name: str,
                batch_size :int = 1,
                max_length :int = 256, 
                device :str = 'cuda:0',
                dtype = torch.float16,
                magicpig_config = None) -> None:

        self.llm = LLM(model_name, batch_size, max_length, device, dtype, magicpig_config)
        

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor, 
            position_ids: Optional[torch.LongTensor] = None
            ):

            if input_ids.shape[1] == 1:
                logits = self.llm.inference(input_ids, position_ids)
            else:
                logits = self.llm.prefill(input_ids, position_ids)
            return logits
    
    
