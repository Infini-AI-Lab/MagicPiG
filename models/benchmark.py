import torch
from magicpig_llama import LLMEngine
import time
import argparse
import torch
import numpy as np
import random
import jsonlines
import json
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
parser = argparse.ArgumentParser()
parser.add_argument('--gen_len', type=int, default=64, help='generation length')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=3500, help='prefix length')
parser.add_argument('--M', type=int, default=4096, help='max length')
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda", 0)
STORAGE = torch.device("cpu")
GEN_LEN = args.gen_len
B = args.B
WARM_UP = 32

with open("magicpig_config.json") as f:
    config = json.load(f)

MODEL_NAME = config["model_path"]

llm = LLMEngine(MODEL_NAME, B, MAX_LEN, DEVICE, DTYPE, config)
with open("../data/data.jsonl") as f:
    d = jsonlines.Reader(f)
    for idx, item in enumerate(d):
        data = item
        break
text = data["input"]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
input_ids = tokenizer.encode(data["input"], return_tensors="pt")
input_ids = input_ids[:,1024:PREFIX_LEN + WARM_UP + GEN_LEN]
input_ids = input_ids.repeat(B, 1)
position_ids = torch.arange(PREFIX_LEN + WARM_UP + GEN_LEN).unsqueeze(0).repeat(B, 1)
input_ids = input_ids.to(DEVICE)
position_ids = position_ids.to(DEVICE)
logits = llm.inference(input_ids=input_ids[:,:PREFIX_LEN], position_ids=position_ids[..., :PREFIX_LEN])
print("Start Decoding")
torch.cuda.synchronize()
for k in range(WARM_UP):
        logits = llm.inference(input_ids=input_ids[..., 128+k:128+k+1], position_ids=position_ids[..., PREFIX_LEN+k:PREFIX_LEN+k+1])
torch.cuda.synchronize()

torch.cuda.synchronize()
t1 = time.time()
for k in range(GEN_LEN):
        logits = llm.inference(input_ids=input_ids[..., WARM_UP+k:WARM_UP+k+1], position_ids=position_ids[..., WARM_UP+PREFIX_LEN+k:WARM_UP+PREFIX_LEN+k+1])
torch.cuda.synchronize()
t2 = time.time()

print("Decoding Latency {:.3f} ms/token".format(1000 * (t2 - t1)/GEN_LEN))
print("Decoding Throughput {:.3f} token/s".format(B * GEN_LEN / (t2 - t1)))


