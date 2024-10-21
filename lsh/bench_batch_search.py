import torch
import argparse
import torch.utils.benchmark as benchmark
import lsh
import random

batch_search_op = lsh.torch_lsh_batch_search_mixing_gqa

def batch_search(bucket_start, bucket_end, bucket, q, results, nnz, bitmask):
    batch_search_op(bucket_start, bucket_end, bucket, q, results, nnz, bitmask, 1)
    bitmask.zero_()
def bench(num_bucket:int, seq_len: int, group_size :int, bsz :int):
    
    
    batch_bucket_start_list = []
    batch_bucket_end_list = []
    batch_bucket_list = []
    batch_q = []
    for _ in range(bsz//4):
        bucket_start_list = []
        bucket_end_list = []
        bucket_list = []
        for _ in range(group_size):
            offset = torch.randint(low=0, high=seq_len, size=(num_bucket - 1,)).to(torch.int)
            offset = torch.sort(offset).values
            
            bucket_start = torch.zeros(size=(num_bucket,)).to(torch.int)
            bucket_start[1:] = offset

            bucket_end = torch.full(size=(num_bucket,), fill_value=seq_len).to(torch.int)
            bucket_end[:-1] = offset

            bucket_start_list.append(bucket_start)
            bucket_end_list.append(bucket_end)

            bucket = torch.randperm(seq_len).to(torch.int)
            bucket_list.append(bucket)
            


        
        
        bucket_start = torch.stack(bucket_start_list).contiguous()
        bucket_end = torch.stack(bucket_end_list).contiguous()
        bucket = torch.stack(bucket_list).contiguous()
        
        
        batch_bucket_start_list.append(bucket_start)
        batch_bucket_end_list.append(bucket_end)
        batch_bucket_list.append(bucket)
        

    batch_bucket_start = torch.stack(batch_bucket_start_list).contiguous()
    batch_bucket_end = torch.stack(batch_bucket_end_list).contiguous()
    batch_bucket = torch.stack(batch_bucket_list).contiguous()

    batch_q = torch.randint(low=0, high=num_bucket, size=(bsz, group_size)).to(torch.int)
    nnz = torch.zeros((bsz,)).to(torch.int)
    results_lsh_cpu = torch.zeros((bsz, seq_len)).to(torch.int)
    bitmask = torch.zeros((bsz, seq_len)).to(torch.uint8)
    
    t_op = benchmark.Timer(
        stmt="batch_search(bucket_start, bucket_end, bucket, q, results, nnz, bitmask)",
        globals={"batch_search": batch_search, "bucket_start": batch_bucket_start, "bucket_end": batch_bucket_end,  "bucket": batch_bucket, "q": batch_q, "results": results_lsh_cpu, "nnz": nnz, "bitmask":bitmask},
        label="batch_search",
        num_threads=32
    )

    print("NUM BUCKETS = {}, SEQ_LEN = {}, GROUP SIZE = {}, BSZ = {}".format(num_bucket, seq_len, group_size, bsz))
    print(t_op.timeit(1))
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--bucket", type=int, default=1024)
    argparser.add_argument("--seq", type=int, default=98304)
    argparser.add_argument("--group", type=int, default=300)
    argparser.add_argument("--bsz", type=int, default=32)
    args = argparser.parse_args()
    bench(args.bucket, args.seq, args.group, args.bsz)

