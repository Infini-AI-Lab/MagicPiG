import torch
from magicpig_llama import LLMEngine
import jsonlines
import json
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
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda", 0)
STORAGE = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default="../data/data32k.json", help='generation files')
args = parser.parse_args()
print(args)

with open(args.path) as f:
    d = jsonlines.Reader(f)
    for idx, item in enumerate(d):
        data = item
        break

with open("magicpig_config.json") as f:
    config = json.load(f)

GEN_LEN = config["gen_len"]
MODEL_NAME = config["model_path"]
text = data["input"]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
input_ids = tokenizer.encode(data["input"], return_tensors="pt")
PREFIX_LEN = input_ids.shape[1]
position_ids = torch.arange(PREFIX_LEN + GEN_LEN).unsqueeze(0)
input_ids = input_ids.to(DEVICE)
position_ids = position_ids.to(DEVICE)
llm = LLMEngine(MODEL_NAME, 1, GEN_LEN + PREFIX_LEN + 128, DEVICE, DTYPE, config)
logits = llm.inference(input_ids=input_ids, position_ids=position_ids[:,:PREFIX_LEN])

generated = input_ids[0].tolist()
for k in range(GEN_LEN):
    input_ids = logits.argmax(dim=-1)
    logits = llm.inference(input_ids=input_ids, position_ids=position_ids[:,PREFIX_LEN + k:PREFIX_LEN + k + 1])
    generated.append(input_ids.item())
    if input_ids.item() in config["eos"]:
        break
text = tokenizer.decode(generated, skip_special_tokens=True)

print(text)

