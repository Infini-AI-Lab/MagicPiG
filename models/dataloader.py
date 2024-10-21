import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_longbench_dummy(seq_len: int, batch_size: int, model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    dataset = load_dataset('THUDM/LongBench', "2wikimqa_e", split='test')
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
            content = examples["context"] 
            return tokenizer(content, return_tensors='pt',max_length=seq_len,padding="max_length",truncation=True)
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['context', 'input', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    input_ids:torch.LongTensor = dataset["input_ids"]
    input_ids.squeeze_(1)
    attention_mask:torch.LongTensor = dataset["attention_mask"]
    attention_mask.squeeze_(1)
    
    real_input = []
    real_mask = []
    for i in range(len(input_ids)):
          if len(real_input) == batch_size: break
          else:
                if (attention_mask[i] == 0).sum() == False:
                      real_input.append(input_ids[i])
                      real_mask.append(attention_mask[i])

    input_ids = torch.stack(real_input, dim=0)
    attention_mask = torch.stack(real_mask, dim=0)
    assert (attention_mask == 0).sum() == False
    
    return input_ids, attention_mask