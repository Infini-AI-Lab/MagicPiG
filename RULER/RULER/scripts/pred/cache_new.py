from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F
class SimCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, K, L, mode = "anns", window = 64, num_qh = 32, num_kh = 8, head_dim = 128, num_layers = 32, device = "cuda", dtype=torch.bfloat16) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.selected_key_cache: List[torch.Tensor] = []
        self.selected_value_cache: List[torch.Tensor] = []
        self.unselected_key_cache: List[torch.Tensor] = []
        self.unselected_value_cache: List[torch.Tensor] = []
        self.prefill_tokens = 0
        self.sampling_prob: torch.Tensor = None
        self.kernel_size = 5
        self.interleave = 8
        self.hash_matrix = None
        self.num_qh = None
        self.num_kh = None
        self.head_dim = None
        self.K = K
        self.L = L
        self.recall = None
        
        self.mode = mode
        self.key_hashcode: List[torch.Tensor] = []
        self.expand_key: List[torch.Tensor] = []
        self.avg_key: List[torch.Tensor] = []
        self.expand_key_norm: List[torch.Tensor] = []
        self.window = window
        self.hash_matrices: List[torch.Tensor] = []
        self.preserve_layer = 2
        
        self.num_qh = num_qh
        self.num_kh = num_kh
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.chunk_size = 8
        self.hash_matrix = [torch.randn((1, self.num_qh, self.head_dim + 1, self.K * self.L), device=self.device, dtype=self.dtype) for _ in range((self.num_layers-1) //  16 + 1)]
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        random_sparse: float = 1.0,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        

        # Update the cache
        if key_states.shape[2] > 1:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefill_tokens += key_states.shape[2]
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.prefill_tokens += key_states.shape[2]
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            key_states = key_states - self.avg_key[layer_idx]
            self.selected_key_cache[layer_idx] = torch.cat([self.selected_key_cache[layer_idx], key_states], dim=-2)
            self.selected_value_cache[layer_idx] = torch.cat([self.selected_value_cache[layer_idx], value_states], dim=-2)
            num_random_cache = int(random_sparse * (self.prefill_tokens - self.window))
            if layer_idx >= 2:
                
                if num_random_cache > 0:
                    q = query_states / query_states.norm(p=2, dim=-1, keepdim=True)
                    
                    q_hashcode = torch.matmul(q, self.hash_matrix[layer_idx // 16][...,:-1,:].to(q.device)).reshape(1, self.num_qh, query_states.shape[2], self.L, self.K).gt(0)
                    
                    q_hashcode = sum((q_hashcode[...,i].int() * int(2**i)) for i in range(self.K))
                    q_hashcode = q_hashcode.to(torch.int16)
                    k_hashcode = self.key_hashcode[layer_idx]
                    
                    
                    mask = (q_hashcode == k_hashcode).int().sum(dim=-1).float()
                   
                    mask = F.max_pool1d(mask, kernel_size=5, stride=1, padding=2)
                    num_activate_tokens = 324
                    token_indices = mask.topk(k=num_activate_tokens, dim=-1).indices
                    
                    token_indices = token_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                    
                    unselected_key_cache = repeat_kv(self.unselected_key_cache[layer_idx], self.num_qh // self.num_kh)
                    unselected_value_cache = repeat_kv(self.unselected_value_cache[layer_idx], self.num_qh // self.num_kh)

                    unselected_key_cache = unselected_key_cache.gather(dim=-2, index=token_indices)
                    unselected_value_cache = unselected_value_cache.gather(dim=-2, index=token_indices)
                    
                    selected_key_cache = repeat_kv(self.selected_key_cache[layer_idx], self.num_qh // self.num_kh)
                    
                    
                    
                    attn_unselected = torch.matmul(query_states, unselected_key_cache.transpose(2,3))
                    attn_unselected = attn_unselected.to(torch.float32)
                    attn_unselected = attn_unselected / math.sqrt(self.head_dim)

                    attn_selected = torch.matmul(query_states, selected_key_cache.transpose(2,3)) / math.sqrt(self.head_dim)
                    attn_selected = attn_selected.to(torch.float32)
                    
                    
                    attn_weights = torch.cat([attn_selected, attn_unselected], dim=-1)
                    
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    return_value = repeat_kv(self.selected_value_cache[layer_idx], self.num_qh //  self.num_kh)
                    return_value = torch.cat([return_value, unselected_value_cache], dim=-2)

                    attn_output = torch.matmul(attn_weights, return_value)
                    
                    return attn_output
                else:
                    return_key = self.selected_key_cache[layer_idx]
                    return_value = self.selected_value_cache[layer_idx]
                    
                    return_key = repeat_kv(return_key, self.num_qh // self.num_kh)
                    return_value = repeat_kv(return_value, self.num_qh // self.num_kh)
                        
                    attn_weights = torch.matmul(query_states, return_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                        
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
                    attn_output = torch.matmul(attn_weights, return_value)
                    return attn_output
            else:
                return_key = repeat_kv(self.key_cache[layer_idx], self.num_qh // self.num_kh)
                return_value = repeat_kv(self.value_cache[layer_idx], self.num_qh // self.num_kh)
                        
                attn_weights = torch.matmul(query_states, return_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                        
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
                attn_output = torch.matmul(attn_weights, return_value)
                return attn_output

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "SimCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def select_kv_cache(self, num_activate_tokens:int, sorted_indices: torch.LongTensor, layer_idx: int, window_size: int, head_dim: int):
        
        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]
        
        select_indices = sorted_indices[...,:num_activate_tokens].unsqueeze(-1).expand(-1,-1,-1,head_dim)
        unselect_indices = sorted_indices[...,num_activate_tokens:].unsqueeze(-1).expand(-1,-1,-1,head_dim)
        
        select_k_cache = k_cache.gather(dim=-2, index=select_indices)
        select_v_cache = v_cache.gather(dim=-2, index=select_indices)
        
        select_k_cache = torch.cat([select_k_cache, k_cache[...,-window_size:,:]], dim = -2)
        select_v_cache = torch.cat([select_v_cache, v_cache[...,-window_size:,:]], dim = -2)
        
        
        unselect_k_cache = k_cache.gather(dim=-2, index=unselect_indices)
        unselect_v_cache = v_cache.gather(dim=-2, index=unselect_indices)
        
        
        self.selected_key_cache.append(select_k_cache)
        self.selected_value_cache.append(select_v_cache)
        
        self.unselected_key_cache.append(unselect_k_cache)
        self.unselected_value_cache.append(unselect_v_cache)
        
        self.avg_key.append(unselect_k_cache.mean(dim=-2, keepdim=True))

        self.selected_key_cache[layer_idx] = self.selected_key_cache[layer_idx] - self.avg_key[layer_idx]
        self.unselected_key_cache[layer_idx] = self.unselected_key_cache[layer_idx] - self.avg_key[layer_idx]
        
        expand_k = repeat_kv(self.unselected_key_cache[layer_idx], self.num_qh // self.num_kh)
        
        expand_k_norm = expand_k.norm(p=2, dim=-1)
        
        expand_k_norm_max = expand_k_norm.max(dim=-1, keepdim=True).values + 1e-5
        
        cat_tensor = torch.sqrt(expand_k_norm_max.pow(2) - expand_k_norm.pow(2))
        self.expand_key_norm.append(expand_k_norm_max)
        cat_tensor = cat_tensor.unsqueeze(-1)
        expand_k = torch.cat([expand_k, cat_tensor], dim=-1)
         
        hash_code = torch.matmul(expand_k, self.hash_matrix[layer_idx // 16].to(expand_k.device)).reshape(1, self.num_qh, expand_k.shape[2], self.L, self.K)

        hash_code = hash_code > 0
        
        hash_code = sum([(hash_code[...,i].int() * int(2**i)) for i in range(self.K)])
        hash_code = hash_code.to(torch.int16)
        self.key_hashcode.append(hash_code)
        