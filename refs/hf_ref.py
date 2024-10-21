from dataloader import load_longbench_dummy, AutoTokenizer
from hf_model_ref import LlamaForCausalLM
import argparse
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--gen_len', type=int, default=128, help='generation length')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=3500, help='prefix length')
parser.add_argument('--M', type=int, default=4096, help='max length')
args = parser.parse_args()
print(args)

PREFIX_LEN = args.P
MAX_LEN = args.M
MODEL_NAME = args.model
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
GEN_LEN = args.gen_len
WARM_UP = 8
B = args.B
SANITY = True
CHECK_STEP = 3
input_ids, mask = load_longbench_dummy(seq_len=PREFIX_LEN + 128, batch_size=B, model_name=MODEL_NAME)
input_ids = input_ids.to(DEVICE)
mask = mask.to(DEVICE)

padding = torch.ones((B, CHECK_STEP)).to(DEVICE)

mask = torch.cat([mask, padding], dim=1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")


with torch.inference_mode():
        hf_llm : LlamaForCausalLM = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, _attn_implementation="eager").to(DEVICE)
        hf_llm = hf_llm.eval()
        hf_llm.config.K = 10
        hf_llm.config.L = 150
        hf_llm.config.cache_mode = "anns_es"
        hf_llm.config.window = 64
        hf_llm.set_sparse_attn(sparse=4, window_size=64, kernel_size=5)
        hf_llm.select_kv(select=True)
        hf_kv_cache = None
        # encoding
        output_hf = hf_llm(input_ids=input_ids[:,:PREFIX_LEN], use_cache=True, past_key_values=hf_kv_cache, attention_mask=mask[:, :PREFIX_LEN])
        logits_hf = output_hf.logits
        hf_kv_cache = output_hf.past_key_values
        print(logits_hf)
        for k in range(CHECK_STEP):
            output_hf = hf_llm(input_ids=input_ids[..., 128+k:128+k+1], use_cache=True, past_key_values=hf_kv_cache, attention_mask=mask[:, :PREFIX_LEN+ k+1])
            logits_hf = output_hf.logits
            hf_kv_cache = output_hf.past_key_values
            print(logits_hf.topk(3))


