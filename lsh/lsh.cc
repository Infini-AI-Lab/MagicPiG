#include<cstdint>
#include<cstdio>
#include<cstring>
#include<iostream>
#include<torch/extension.h>
#include<assert.h>
#include <cassert>
#include <algorithm>

int lsh_bucket_search_mixing(
const int *bucket_start, 
const int *bucket_end, 
const int q, 
const int *bucket, 
int *result,
uint8_t *bitmask,
int threshold){

    int start = bucket_start[q];
    int end = bucket_end[q];
    int nnz = 0;
    for (int i = start; i < end; ++i){
        int idx = bucket[i];
        if (bitmask[idx] == 0){
            bitmask[idx] = 1;
        }
        else if (bitmask[idx] == 1)
        {
            bitmask[idx] = 2;
            result[nnz] = idx;
            nnz ++;
        }
        
    }
    return nnz;
}


int lsh_group_search_mixing(
    const int *bucket_start, 
    const int *bucket_end, 
    const int *q, 
    const int *bucket, 
    int *result, 
    uint8_t *bitmask,
    int num_tables, int num_buckets, int seq_len, int threshold){

    int offset = 0; 
    for (uint32_t io = 0; io < num_tables; ++io){
        //std::cout << "table id " << io << " offset " << offset << std::endl;
        int shift = lsh_bucket_search_mixing(bucket_start + io * num_buckets, bucket_end + io * num_buckets, q[io], bucket + io * seq_len, result + offset, bitmask, threshold);
        
        offset = offset + shift;
    }
    return offset;
}


void batched_lsh_group_search_mixing(
const int *bucket_start, 
const int *bucket_end, 
const int *q, 
const int *bucket, 
int *result, 
uint8_t *bitmask,
int * nnz, int bsz, int num_tables, int num_buckets, int seq_len, int result_offset, int threshold){


    
    //#pragma omp parallel for schedule(static, 1) num_threads(96)
    for(int i = 0; i < bsz; ++i){
        nnz[i] = lsh_group_search_mixing(
            bucket_start + i * (num_tables * num_buckets), 
            bucket_end + i * (num_tables * num_buckets), 
            q + i * num_tables, 
            bucket + i * (num_tables * seq_len), 
            result + i * result_offset, 
            bitmask + i * seq_len,
            num_tables, num_buckets, seq_len, threshold);
    }
    
    
}

void batched_lsh_group_search_mixing_gqa(
const int *bucket_start, 
const int *bucket_end, 
const int *q, 
const int *bucket, 
int *result, 
uint8_t *bitmask,
int * nnz, int bsz, int num_tables, int num_buckets, int seq_len, int result_offset, int threshold, int group){

    #pragma omp parallel for schedule(static, 1) num_threads(96)
    for(int i = 0; i < bsz; ++i){
        nnz[i] = lsh_group_search_mixing(
            bucket_start + (i/group) * (num_tables * num_buckets), 
            bucket_end + (i/group) * (num_tables * num_buckets), 
            q + i * num_tables, 
            bucket + (i/group) * (num_tables * seq_len), 
            result + i * result_offset, 
            bitmask + i * seq_len,
            num_tables, num_buckets, seq_len, threshold);
    }
     
}

void build_tables(
const int* values,
int* start,
int* end,
int num_heads,
int num_buckets,
int num_tables,
int seq_len,
int req_id
){
    int *m_start = start + req_id * num_heads * num_tables * num_buckets;
    int *m_end = end + req_id * num_heads * num_tables * num_buckets;
    #pragma omp parallel for num_threads(96)
    for(int i = 0;  i < num_heads; ++i){
        const int * v_i = values + i * (num_tables * seq_len);
        int *ms_i = m_start + i * (num_tables * num_buckets);
        int *me_i = m_end+ i * (num_tables * num_buckets);
        for(int j=0; j < num_tables; ++j)
            {
            
            const int * v_ij = v_i + j * seq_len;
            int *ms_ij = ms_i + j * num_buckets;
            int *me_ij = me_i + j * num_buckets;
            for(int k=0; k<seq_len; ++k){
                const int v = v_ij[k];
                if(me_ij[v]==0){
                    ms_ij[v] = k;
                    me_ij[v] = k+1;
                }else{
                    me_ij[v] = me_ij[v] + 1;
                }
            }
            }
    }
}
void torch_build_tables(torch::Tensor values, torch::Tensor start, torch::Tensor end,
                                        int req_id) {

    int num_heads = values.size(0);
    int num_buckets = start.size(2);
    int num_tables = values.size(1);
    int seq_len = values.size(2);
    build_tables(
        static_cast<int *>(values.data_ptr()),
        static_cast<int *>(start.data_ptr()),
        static_cast<int *>(end.data_ptr()),
        num_heads,
        num_buckets,
        num_tables,
        seq_len,
        req_id
    );
}


int torch_lsh_bucket_search_mixing(torch::Tensor bucket_start, torch::Tensor bucket_end, torch::Tensor bucket,
                                        int q, torch::Tensor result, torch::Tensor bitmask, int threshold) {

            return lsh_bucket_search_mixing(
                static_cast<int *>(bucket_start.data_ptr()),
                static_cast<int *>(bucket_end.data_ptr()),
                q,
                static_cast<int *>(bucket.data_ptr()),
                static_cast<int *>(result.data_ptr()),
                static_cast<uint8_t *>(bitmask.data_ptr()),
                threshold
            );
}


int torch_lsh_group_search_mixing(torch::Tensor bucket_start, torch::Tensor bucket_end, torch::Tensor bucket,
                                        torch::Tensor q, torch::Tensor result, torch::Tensor bitmask, int threshold) {
            
            int num_tables = bucket_start.size(0);
            int num_buckets = bucket_start.size(1);
            int seq_len = bucket.size(1);
            return lsh_group_search_mixing(
                static_cast<int *>(bucket_start.data_ptr()),
                static_cast<int *>(bucket_end.data_ptr()),
                static_cast<int *>(q.data_ptr()),
                static_cast<int *>(bucket.data_ptr()),
                static_cast<int *>(result.data_ptr()),
                static_cast<uint8_t *>(bitmask.data_ptr()),
                num_tables,
                num_buckets,
                seq_len,
                threshold
            );

}


void torch_lsh_batch_search_mixing(torch::Tensor bucket_start, torch::Tensor bucket_end, torch::Tensor bucket,
                                        torch::Tensor q, torch::Tensor result, torch::Tensor nnz, torch::Tensor bitmask, int threshold) {
            
            int bsz = bucket_start.size(0);
            int num_tables = bucket_start.size(1);
            int num_buckets = bucket_start.size(2);
            int seq_len = bucket.size(2);
            int result_offset = result.size(1);
            batched_lsh_group_search_mixing(
                static_cast<int *>(bucket_start.data_ptr()),
                static_cast<int *>(bucket_end.data_ptr()),
                static_cast<int *>(q.data_ptr()),
                static_cast<int *>(bucket.data_ptr()),
                static_cast<int *>(result.data_ptr()),
                static_cast<uint8_t *>(bitmask.data_ptr()),
                static_cast<int *>(nnz.data_ptr()),
                bsz,
                num_tables,
                num_buckets,
                seq_len,
                result_offset,
                threshold
            );

}

void torch_lsh_batch_search_mixing_gqa(torch::Tensor bucket_start, torch::Tensor bucket_end, torch::Tensor bucket,
                                        torch::Tensor q, torch::Tensor result, torch::Tensor nnz, torch::Tensor bitmask, int threshold) {
            
            int bsz = q.size(0);
            int group = static_cast<int>(q.size(0) / bucket_start.size(0));
            int num_tables = bucket_start.size(1);
            int num_buckets = bucket_start.size(2);
            int seq_len = bucket.size(2);
            int result_offset = result.size(1);
            batched_lsh_group_search_mixing_gqa(
                static_cast<int *>(bucket_start.data_ptr()),
                static_cast<int *>(bucket_end.data_ptr()),
                static_cast<int *>(q.data_ptr()),
                static_cast<int *>(bucket.data_ptr()),
                static_cast<int *>(result.data_ptr()),
                static_cast<uint8_t *>(bitmask.data_ptr()),
                static_cast<int *>(nnz.data_ptr()),
                bsz,
                num_tables,
                num_buckets,
                seq_len,
                result_offset,
                threshold,
                group
            );

}



PYBIND11_MODULE(lsh, m) {

m.def("torch_lsh_bucket_search_mixing", &torch_lsh_bucket_search_mixing,
        "lsh search for a single bucket (CPU Version)");

m.def("torch_lsh_group_search_mixing", &torch_lsh_group_search_mixing,
        "lsh search for a bucket group (CPU Version)");

m.def("torch_lsh_batch_search_mixing", &torch_lsh_batch_search_mixing,
        "batched lsh search (CPU Version)");

m.def("torch_lsh_batch_search_mixing_gqa", &torch_lsh_batch_search_mixing_gqa,
        "batched lsh search (CPU Version)");

m.def("torch_build_tables", &torch_build_tables,
        "build tables with req id");
}
