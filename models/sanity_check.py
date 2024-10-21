import torch
from magicpig_llama import LLMEngine

from dataloader import load_longbench_dummy
import argparse
import torch
import numpy as np
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--gen_len', type=int, default=64, help='generation length')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=3500, help='prefix length')
parser.add_argument('--M', type=int, default=4096, help='max length')
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
MODEL_NAME = args.model
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda", 0)
STORAGE = torch.device("cpu")
GEN_LEN = args.gen_len
B = args.B
CHECK_STEP = 3
WARM_UP = 32
T = 1024


llm = LLMEngine(MODEL_NAME, B, MAX_LEN, DEVICE, DTYPE)
position_ids = torch.arange(PREFIX_LEN + WARM_UP + GEN_LEN).unsqueeze(0).repeat(B, 1)
input_ids,mask= load_longbench_dummy(seq_len=PREFIX_LEN + WARM_UP + GEN_LEN, batch_size=1, model_name=MODEL_NAME)
input_ids = input_ids.to(DEVICE)
position_ids = position_ids.to(DEVICE)
input_ids = input_ids.repeat(B, 1)
logits = llm.inference(input_ids=input_ids[:,:PREFIX_LEN], position_ids=position_ids[..., :PREFIX_LEN])
print(logits)
for k in range(CHECK_STEP):
        logits = llm.inference(input_ids=input_ids[..., 128+k:128+k+1], position_ids=position_ids[..., PREFIX_LEN+k:PREFIX_LEN+k+1])
        print(logits.topk(3))