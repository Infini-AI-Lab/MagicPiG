from transformers import LlamaConfig
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
import math
from torch import nn
from utils import repeat_kv
import lsh
build_tables = lsh.torch_build_tables
batch_search_op = lsh.torch_lsh_batch_search_mixing_gqa
import gather_gemv_cpu
batch_sparse_attention = gather_gemv_cpu.batch_sparse_attention
batch_transform = gather_gemv_cpu.batch_transform
batch_softmax = gather_gemv_cpu.batch_softmax
batch_wv = gather_gemv_cpu.batch_sparse_attention_wv
class KV_Cache:
    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        cache_layers :list[int] = [0, 16],
        device_budget :int = 68,
        generate_buffer :int = 256,
        K = 10,
        L = 150,
        mem_efficient = True,
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache_layers = cache_layers
        self.device_budget = device_budget
        self.generate_buffer = generate_buffer
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.mem_efficient = mem_efficient
        self.k_cache :list[torch.Tensor] = [torch.zeros(
            batch_size * config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device="cpu",
            dtype=self.dtype
        ) for _ in range(config.num_hidden_layers)]

        self.v_cache :list[torch.Tensor]= [torch.zeros(
            batch_size *config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device="cpu",
            dtype=self.dtype
        ) for _ in range(config.num_hidden_layers)]

        self.gpu_k_cache :list[torch.Tensor] = [torch.zeros(
            batch_size *config.num_key_value_heads,
            (self.device_budget + self.generate_buffer) if (i not in self.cache_layers) else 0,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        ) for i in range(config.num_hidden_layers)]

        self.gpu_v_cache :list[torch.Tensor]= [torch.zeros(
            batch_size *config.num_key_value_heads,
            (self.device_budget + self.generate_buffer) if (i not in self.cache_layers) else 0,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        ) for i in range(config.num_hidden_layers)]

        self.avg_key :list[torch.Tensor] = [torch.zeros(
            batch_size *config.num_key_value_heads,
            1,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        ) for i in range(config.num_hidden_layers)]

        self.expand_k_norm :list[torch.Tensor] = [torch.ones(
            batch_size *config.num_attention_heads,
            max_length,
            device="cpu",
            dtype=torch.float32
        ) for i in range(config.num_hidden_layers)
        ]

        self.kv_offset = [0 for _ in range(self.num_layers)]
        self.gpu_kv_offset = [0 for _ in range(self.num_layers)]

        self.device_k_cache = torch.zeros(
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.device_v_cache = torch.zeros(
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.K = K
        self.L = L
        self.num_bucket = int(2**self.K)
        self.hash_matrix: list[torch.Tensor] = [torch.randn((1, self.num_key_value_heads, self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype) for _ in range((self.num_layers-1) //  16 + 1)]
        self.binary_pack = [int(2**i) for i in range(self.K)]
        self.binary_pack = torch.Tensor(self.binary_pack).unsqueeze(1).to(self.device)

        self.layer_batch_start = [
            torch.zeros(
            self.batch_size * self.num_key_value_heads,
            self.L,
            self.num_bucket,
            device="cpu",
            dtype=torch.int32)
            for _ in range(self.config.num_hidden_layers)
        ]

        self.layer_batch_end = [
            torch.zeros(
            self.batch_size * self.num_key_value_heads,
            self.L,
            self.num_bucket,
            device="cpu",
            dtype=torch.int32)
            for _ in range(self.config.num_hidden_layers)
        ]

        self.layer_batch_bucket = [
            torch.zeros(
            self.batch_size * self.num_key_value_heads,
            self.L,
            self.max_length,
            device="cpu",
            dtype=torch.int32) 
            for _ in range(self.config.num_hidden_layers)
        ]

        self.nnz = torch.zeros((self.batch_size * self.num_heads,)).to(torch.int32)
        self.real_nnz = torch.ones((self.batch_size * self.num_heads,)).to(torch.int32) * 6144
        self.results_lsh_cpu = torch.zeros((self.batch_size * self.num_heads, self.max_length)).to(torch.int32)
        self.bitmask = torch.zeros((self.batch_size * self.num_heads, self.max_length)).to(torch.uint8)
        self.threshold = 2

        self.attn_weight = torch.zeros((self.batch_size * self.num_heads, self.max_length), dtype=torch.float32)
        self.max_value_expsum = torch.ones((2, self.batch_size * self.num_heads)).to(torch.float32).pin_memory()
        self.output_cuda = torch.zeros((self.batch_size * self.num_heads, self.head_dim), dtype=torch.bfloat16).to(self.device)
        self.max_value_expsum_cuda = torch.ones((2, self.batch_size * self.num_heads)).to(torch.float32).to(self.device)
        self.output = torch.zeros((self.batch_size * self.num_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()

        self.pinned_hashcode = torch.zeros((self.batch_size * self.num_heads, self.L), dtype=torch.int32).pin_memory()
        self.pinned_query = torch.zeros((self.batch_size * self.num_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.query_norm = torch.ones((self.batch_size * self.num_heads,)).float()
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):
        

        current_len = new_k_cache.shape[-2]
        past_len = self.kv_offset[layer_idx]
        self.k_cache[layer_idx][...,past_len:current_len+past_len,:].copy_(new_k_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
        self.v_cache[layer_idx][...,past_len:current_len+past_len,:].copy_(new_v_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
        k = torch.narrow(self.k_cache[layer_idx], dim=1, start=0, length=current_len+past_len)
        v = torch.narrow(self.v_cache[layer_idx], dim=1, start=0, length=current_len+past_len)
        self.kv_offset[layer_idx] = current_len + past_len

        return k,v
    
    def prefill(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            start: int,
            end: int
            ):
        
        self.device_k_cache[...,start:end,:] = new_k_cache
        self.device_v_cache[...,start:end,:] = new_v_cache
        
        
        k = torch.narrow(self.device_k_cache, dim=2, start=0, length=end)
        v = torch.narrow(self.device_v_cache, dim=2, start=0, length=end)
        return k,v
    
    def offload_kv_cache(self, layer_idx:int, offset:int, window_size: int = None):
        
        
        if layer_idx not in self.cache_layers:
            
            if offset <= self.device_budget:
                self.gpu_k_cache[layer_idx][:,:offset,:].copy_(self.device_k_cache[:,:,:offset,:].reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
                self.gpu_v_cache[layer_idx][:,:offset,:].copy_(self.device_v_cache[:,:,:offset,:].reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
                self.gpu_kv_offset[layer_idx] = offset
            else:
                
                select_k_cache = self.device_k_cache[...,:self.device_budget - window_size,:]
                select_v_cache = self.device_v_cache[...,:self.device_budget - window_size,:]


                select_k_cache = torch.cat([select_k_cache, self.device_k_cache[...,offset-window_size:offset,:]], dim = -2)
                select_v_cache = torch.cat([select_v_cache, self.device_v_cache[...,offset-window_size:offset,:]], dim = -2)

        
                unselect_k_cache = self.device_k_cache[...,self.device_budget - window_size:offset-window_size,:]
                unselect_v_cache = self.device_v_cache[...,self.device_budget - window_size:offset-window_size,:]

                unselect_k_cache = unselect_k_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim)
                self.avg_key[layer_idx].copy_(unselect_k_cache.mean(dim=-2, keepdim=True))
                unselect_k_cache = (unselect_k_cache - self.avg_key[layer_idx])
                
                self.gpu_k_cache[layer_idx][...,:self.device_budget,:] = (select_k_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim) - self.avg_key[layer_idx])
                self.gpu_v_cache[layer_idx][...,:self.device_budget,:] = select_v_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim)
                self.gpu_kv_offset[layer_idx] = self.device_budget
                
                self.k_cache[layer_idx][...,:offset - self.device_budget,:] = unselect_k_cache
                self.v_cache[layer_idx][...,:offset - self.device_budget,:] = unselect_v_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim)
                self.kv_offset[layer_idx] = offset - self.device_budget
                
                unselect_k_cache = unselect_k_cache.reshape(self.batch_size, self.num_key_value_heads, -1, self.head_dim)
                k_norm = unselect_k_cache.norm(dim=-1, p=2).unsqueeze(-2).repeat(1, 1, self.num_key_value_groups, 1)
                k_norm = k_norm.reshape(self.batch_size * self.num_heads, -1)
                self.expand_k_norm[layer_idx][:,:k_norm.shape[-1]].copy_(k_norm)
                
                hash_code = torch.matmul(unselect_k_cache, self.hash_matrix[layer_idx // 16]).reshape(self.batch_size, self.num_key_value_heads, -1, self.L, self.K)
                hash_code = hash_code > 0
                hash_code = hash_code.to("cpu")
                hash_code = sum([(hash_code[...,i].int() * int(2**i)) for i in range(self.K)])
                hash_code = hash_code.transpose(2,3).contiguous()
                
                hash_code = hash_code.reshape(self.batch_size*self.num_key_value_heads, self.L, -1)
                
                sorted_hash_values, sorted_hash_indices = hash_code.sort()
                sorted_hash_indices = sorted_hash_indices.contiguous()
                sorted_hash_values = sorted_hash_values.contiguous()
                sorted_hash_values = sorted_hash_values.to("cpu")
                build_tables(sorted_hash_values, self.layer_batch_start[layer_idx], self.layer_batch_end[layer_idx], 0)
                self.layer_batch_bucket[layer_idx][...,:self.kv_offset[layer_idx]].copy_(sorted_hash_indices.int())


        else:
            self.k_cache[layer_idx].copy_(self.device_k_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
            self.v_cache[layer_idx].copy_(self.device_v_cache.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
            self.kv_offset[layer_idx] = offset


    def clear(self):

        for i in range(self.num_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()
        self.kv_offset = [0 for _ in range(self.num_layers)]
        
    def prepare_for_decode(self):
        self.device_k_cache = None
        self.device_v_cache = None
        for i in self.cache_layers:
            self.k_cache[i] = self.k_cache[i].to(self.device)
            self.v_cache[i] = self.v_cache[i].to(self.device)
    @torch.inference_mode()
    def attention_compute(self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        
        if layer_idx in self.cache_layers:
            if self.mem_efficient:
                key_states, value_states = self.update_kv_cache(key_states, value_states, layer_idx)
                bsz, _, q_len, _ = query_states.shape
                query_states = query_states.reshape(self.batch_size * self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
                
                attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                hidden_states = torch.matmul(attn_weights, value_states)
                
                hidden_states = hidden_states.reshape(bsz, self.num_heads, q_len, -1)
                hidden_states = hidden_states.transpose(1, 2).contiguous()
                hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
                return hidden_states
            else:
                key_states, value_states = self.update_kv_cache(key_states, value_states, layer_idx)
                key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, -1, self.head_dim)
                value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, -1, self.head_dim)
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                bsz, _, q_len, _ = query_states.shape
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                hidden_states = torch.matmul(attn_weights, value_states)
                hidden_states = hidden_states.transpose(1, 2).contiguous()
                hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
                return hidden_states
        
        else:

            norm_q = query_states.reshape(self.batch_size, self.num_heads, 1, self.head_dim)
            norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True)  
            norm_q = norm_q.reshape(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            norm_q = norm_q.transpose(0,1).reshape(self.num_key_value_heads, self.batch_size * self.num_key_value_groups, self.head_dim)
            q_hashcode = torch.matmul(norm_q, self.hash_matrix[layer_idx // 16]).gt(0)
            q_hashcode = q_hashcode.reshape(self.num_key_value_heads, self.batch_size, self.num_key_value_groups, self.K * self.L).transpose(0,1).contiguous()
            q_hashcode = q_hashcode.reshape(self.batch_size * self.num_heads * self.L, self.K).float()
            q_hashcode = torch.matmul(q_hashcode, self.binary_pack).int()
            q_hashcode = q_hashcode.reshape(self.batch_size * self.num_heads, self.L)
                        
            self.pinned_hashcode.copy_(q_hashcode)
            self.pinned_query.copy_(query_states.reshape(self.batch_size * self.num_heads, self.head_dim))
            current_len = key_states.shape[-2]
            past_len = self.gpu_kv_offset[layer_idx]
            key_states = key_states.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim)
            key_states = key_states - self.avg_key[layer_idx]
            self.gpu_k_cache[layer_idx][...,past_len:current_len+past_len,:].copy_(key_states.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
            self.gpu_v_cache[layer_idx][...,past_len:current_len+past_len,:].copy_(value_states.reshape(self.batch_size * self.num_key_value_heads, -1, self.head_dim))
            self.gpu_kv_offset[layer_idx] = current_len + past_len
            key_states = torch.narrow(self.gpu_k_cache[layer_idx], dim=1, start=0, length=current_len+past_len)
            value_states = torch.narrow(self.gpu_v_cache[layer_idx], dim=1, start=0, length=current_len+past_len)


            bsz, _, q_len, _ = query_states.shape
            query_states = query_states.reshape(self.batch_size * self.num_key_value_heads, q_len * self.num_key_value_groups, self.head_dim)
            
            gpu_attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)).float() / math.sqrt(self.head_dim)

            gpu_max_attn_weight = torch.max(gpu_attn_weights, dim=-1, keepdim=True).values
            gpu_attn_weights -= gpu_max_attn_weight
            gpu_attn_weights.exp_()
            gpu_attn_weight_sum = torch.sum(gpu_attn_weights, dim=-1,keepdim=True)
            gpu_attn_weights = (gpu_attn_weights / gpu_attn_weight_sum).to(query_states.dtype)
            gpu_hidden_states = torch.matmul(gpu_attn_weights, value_states)
            
            key_states = self.k_cache[layer_idx]
            value_states = self.v_cache[layer_idx]
            batch_search_op(
                            self.layer_batch_start[layer_idx], 
                            self.layer_batch_end[layer_idx], 
                            self.layer_batch_bucket[layer_idx], 
                            self.pinned_hashcode, 
                            self.results_lsh_cpu, 
                            self.nnz, self.bitmask, self.threshold)
            
            self.bitmask.zero_()
            batch_sparse_attention(self.pinned_query, key_states, self.results_lsh_cpu, self.attn_weight, self.nnz)
            batch_transform(self.attn_weight, self.nnz, self.pinned_query.norm(p=2, dim=-1).float(), self.expand_k_norm[layer_idx], self.K, self.L, math.sqrt(self.head_dim), self.results_lsh_cpu)
            
            batch_softmax(self.attn_weight, self.nnz, self.max_value_expsum)
            self.max_value_expsum_cuda.copy_(self.max_value_expsum, non_blocking=True)
            batch_wv(value_states, self.results_lsh_cpu, self.attn_weight, self.output, self.nnz)
            self.output_cuda.copy_(self.output, non_blocking=True)

            cpu_max_attn_weight = self.max_value_expsum_cuda[0]
            cpu_attn_weight_sum = self.max_value_expsum_cuda[1]
                
            cpu_max_attn_weight.unsqueeze_(-1)
            cpu_attn_weight_sum.unsqueeze_(-1)
            cpu_hidden_states = self.output_cuda
            gpu_max_attn_weight = gpu_max_attn_weight.reshape(self.batch_size * self.num_heads, 1)
            gpu_attn_weight_sum = gpu_attn_weight_sum.reshape(self.batch_size * self.num_heads, 1)
            gpu_hidden_states = gpu_hidden_states.reshape(self.batch_size * self.num_heads, self.head_dim)
                
            s0 = (cpu_attn_weight_sum / torch.exp(gpu_max_attn_weight))
            s1 = (gpu_attn_weight_sum / torch.exp(cpu_max_attn_weight))
            w0 = (s0 / (s0 + s1)).to(query_states.dtype)
            w1 = (s1 / (s0 + s1)).to(query_states.dtype)
            hidden_states = (w0 * cpu_hidden_states + w1 * gpu_hidden_states)
            hidden_states = hidden_states.reshape(bsz, self.num_heads, q_len, -1)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)

            return hidden_states
