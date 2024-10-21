#include "fbgemm/FbgemmConvert.h"
#include <immintrin.h>
#include <torch/extension.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
using namespace fbgemm;

float softmax_fp32_impl(float *score, const int64_t length, const int64_t nnz) { 

  assert(score);
  float m = *std::max_element(score,score+nnz);
  
  float exp_sum = 0.f;
  for (int i = 0; i < nnz; ++i){
     score[i] = expf32(score[i] - m);
     exp_sum += score[i];
  }
  __m512 sum = _mm512_set1_ps(exp_sum);
  for (int i = 0; i < nnz / 16; ++i){
      __m512 s_i = _mm512_loadu_ps(score + 16 * i);
      s_i = _mm512_div_ps(s_i, sum);
      _mm512_storeu_ps(score + 16 * i, s_i);
  }
  memset(score+nnz, 0, sizeof(float) * (length - nnz));
  return m;
}


void batch_softmax_fp32_impl(
float *batch_score, 
const int *batch_nnz,
float *batch_max_value_expsum,
const int length,  
const int batch_size) { 


  float *batch_max_value = batch_max_value_expsum;
  float *batch_exp_sum = batch_max_value_expsum + batch_size;
  #pragma omp parallel for schedule(static,1) num_threads(96)
  for(int j = 0; j < batch_size; ++j){

    float *score_j = batch_score + j * length;
    int nnz_j = batch_nnz[j];
    int nnz_j_16 = (nnz_j % 16 == 0) ? (nnz_j): (nnz_j + 16 - nnz_j % 16);
    float m = *std::max_element(score_j,score_j+nnz_j);
    float exp_sum = 0.f;
    for (int i = 0; i < nnz_j; ++i){
      score_j[i] = expf32(score_j[i] - m);
      exp_sum += score_j[i];
    }
    __m512 sum = _mm512_set1_ps(exp_sum);
    for (int i = 0; i < nnz_j_16 / 16; ++i){
        __m512 s_i = _mm512_loadu_ps(score_j + 16 * i);
        s_i = _mm512_div_ps(s_i, sum);
        _mm512_storeu_ps(score_j + 16 * i, s_i);
    }
    batch_max_value[j] = m;
    batch_exp_sum[j] = exp_sum;
  }
  
  
}


void batch_transform_fp32_impl(
float *batch_score, 
const int *batch_nnz,
float *q_norm,
float *k_norm,
const int k,
const int l,
const float sqrt_dim,
const int length,  
const int batch_size,
const int *indices) { 

  #pragma omp parallel for schedule(static,1) num_threads(96)
  for (int i = 0; i < batch_size; ++i)
  {
    int nnz = batch_nnz[i];
    float *score_i = batch_score + i * length;
    const int *head_ind = indices + i * length; 
    float *head_k_norm = k_norm + i * length; 
    //float norm = (q_norm[i] * k_norm[i]);
    for(int j = 0; j < nnz; ++j){
        
        const int ind = head_ind[j];
        float norm = (q_norm[i] * head_k_norm[ind]);
        float theta = acosf(score_i[j] / norm);
        float proba = (1 - theta / M_PI);
        float p = powf(proba, k);
        float q = 1 - p;
        // float w = 1 - powf(q, l) - l * powf(q, l-1) * p;
        float w = 1  - powf(q, l-1) * (l * p + q);
        score_i[j] = score_i[j] / sqrt_dim - logf(w + 1e-4);
    }


  }
}


void batch_fuse_transform_fp32_impl(
float *batch_score, 
const int *batch_nnz,
float *q_norm,
float *k_norm,
const int k,
const int l,
const float sqrt_dim,
const int length,  
const int batch_size) { 

  #pragma omp parallel for schedule(static,1) num_threads(96)
  for (int i = 0; i < batch_size; ++i)
  {
    int nnz = batch_nnz[i];
    float *score_i = batch_score + i * length;
    float norm = (q_norm[i] * k_norm[i]);
    for(int j = 0; j < nnz; ++j){
        
        float theta = acosf(score_i[j] / norm);
        float proba = (1 - theta / M_PI);
        proba = 1 - powf(proba, k);
        proba = 1 - powf(proba, l);
        score_i[j] = score_i[j] / sqrt_dim - logf(proba + 1e-4);
    }  


  }
}

void gather_gemv_inner_product_bf16_impl(const bfloat16 *W, const int32_t *ind,
                                         const float *x, bfloat16 *y,
                                         const int64_t M, const int64_t N,
                                         const int64_t nnz) {
  
  for (uint32_t io = 0; io < (nnz / 16); ++io) {
    float y_f32[16];
    for (uint32_t ii = 0; ii < 16; ++ii) {
      y_f32[ii] = 0.f;
    }
    for (uint32_t jo = 0; jo < (N / 16); ++jo) {
      __m512 x_tile = _mm512_loadu_ps(x + jo * 16);
      for (uint32_t ii = 0; ii < 16; ++ii) {
        const uint32_t i = io * 16 + ii;
        const uint32_t row = ind[i];
        float w_f32[16];
        Bfloat16ToFloat_avx512(W + row * N + jo * 16, w_f32, 16);
        __m512 w_tile = _mm512_loadu_ps(w_f32);
        __m512 wx = _mm512_mul_ps(w_tile, x_tile);
        float sum = _mm512_reduce_add_ps(wx);
        y_f32[ii] += sum;
      }
    }
    FloatToBfloat16_avx512(y_f32, y + io * 16, 16);
  }
}

void gather_gemv_inner_product_fp32_impl(const bfloat16 *W, const int32_t *ind,
                                         const float *x, float *y,
                                         const int64_t M, const int64_t N,
                                         const int64_t nnz) {
  
  memset(y, 0, sizeof(float) * nnz);
  float w_f32[16];
  int64_t nnz_16 = (nnz % 16 == 0) ? (nnz): (nnz + 16 - nnz % 16);

  for (uint32_t io = 0; io < (nnz_16 / 16); ++io) {
    
    for (uint32_t jo = 0; jo < (N / 16); ++jo) {
      __m512 x_tile = _mm512_loadu_ps(x + jo * 16);
      for (uint32_t ii = 0; ii < 16; ++ii) {
        const uint32_t i = io * 16 + ii;
        const uint32_t row = ind[i];
        Bfloat16ToFloat_avx512(W + row * N + jo * 16, w_f32, 16);
        __m512 w_tile = _mm512_loadu_ps(w_f32);
        __m512 wx = _mm512_mul_ps(w_tile, x_tile);
        float sum = _mm512_reduce_add_ps(wx);
        y[io * 16 + ii] += sum;
      }
    }

  }
}


void batched_gather_gemv_inner_product_bf16_impl(const bfloat16 *W,
                                         const float *x, bfloat16 *y, int32_t *ind, int32_t *nnz,
                                         const int64_t BSZ, const int64_t M, const int64_t N, const int64_t ind_offset) {
      
      #pragma omp parallel for schedule(static,1) num_threads(96)
      for (int i = 0; i < BSZ; ++i){
        gather_gemv_inner_product_bf16_impl(
            W + i * M * N,
            ind + i * ind_offset,
            x + i * N,
            y + i * ind_offset,
            M,
            N,
            nnz[i]
        );
      }
}

void batched_gather_gemv_inner_product_fp32_impl(const bfloat16 *W,
                                         const float *x, float *y, int32_t *ind, int32_t *nnz,
                                         const int64_t BSZ, const int64_t M, const int64_t N, const int64_t ind_offset) {
      
      #pragma omp parallel for schedule(static,1) num_threads(96)
      for (int i = 0; i < BSZ; ++i){
        gather_gemv_inner_product_fp32_impl(
            W + i * M * N,
            ind + i * ind_offset,
            x + i * N,
            y + i * ind_offset,
            M,
            N,
            nnz[i]
        );
      }
}

void sparse_attention_qk_fp32_impl(const bfloat16 *key, const int32_t *ind,
                                         const float *query, float *attn_weight,
                                         const int64_t SEQ_LEN, const int64_t HEAD_DIM,
                                         const int64_t nnz) {
  
  memset(attn_weight, 0, sizeof(float) * nnz);
  float w_f32[16];
  int64_t nnz_16 = (nnz % 16 == 0) ? (nnz): (nnz + 16 - nnz % 16);
  
  for (uint32_t io = 0; io < (nnz_16 / 16); ++io) {
    
    for (uint32_t jo = 0; jo < (HEAD_DIM / 16); ++jo) {
      __m512 x_tile = _mm512_loadu_ps(query + jo * 16);
      for (uint32_t ii = 0; ii < 16; ++ii) {
        const uint32_t i = io * 16 + ii;
        const uint32_t row = ind[i];
        Bfloat16ToFloat_avx512(key + row * HEAD_DIM + jo * 16, w_f32, 16);
        __m512 w_tile = _mm512_loadu_ps(w_f32);
        __m512 wx = _mm512_mul_ps(w_tile, x_tile);
        float sum = _mm512_reduce_add_ps(wx);
        attn_weight[io * 16 + ii] += sum;
      }
    }

  }
}

void sparse_attention_wv_bf16_impl(const bfloat16 *value, const int32_t *ind,
                                         const float *attn_weight, bfloat16 *output,
                                         const int64_t SEQ_LEN, const int64_t HEAD_DIM,
                                         const int64_t nnz) {
    float y_f32[16];
    for (uint32_t io = 0; io < (HEAD_DIM / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);

        for(uint32_t jo = 0; jo < nnz; ++jo)
          {
            const uint32_t row = ind[jo];
            __m512 x_j = _mm512_set1_ps(attn_weight[jo]);
            float w_f32[16];
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + io * 16, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, output + io * 16, 16);
    }
                                         }

void batch_sparse_attention_wv_impl(
const bfloat16 *value,
const float *attn_weight, 
bfloat16 *output, 
int32_t *ind, 
int32_t *nnz,
const int64_t NUM_Q_HEAD, 
const int64_t GROUP,
const int64_t SEQ_LEN, 
const int64_t HEAD_DIM) {
      
      #pragma omp parallel for schedule(static,1) num_threads(96)
      for (int i = 0; i < NUM_Q_HEAD; ++i){
        sparse_attention_wv_bf16_impl(
            value + (i/GROUP) * SEQ_LEN * HEAD_DIM,
            ind + i * SEQ_LEN,
            attn_weight + i * SEQ_LEN,
            output + i * HEAD_DIM,
            SEQ_LEN,
            HEAD_DIM,
            nnz[i]
        );
      }
}

void batch_sparse_attention_wv(
  torch::Tensor value, 
  torch::Tensor indices,
  torch::Tensor attn_weight, torch::Tensor attn_output, torch::Tensor nnz) {
  // value: (NUM_Q_HEAD, SEQ_LEN , HEAD_DIM)
  // attn_weight: (NUM_Q_HEAD, SEQ_LEN )
  // attn_output: (NUM_Q_HEAD, HEAD_DIM)
  // indices: (NUM_Q_HEAD, SEQ_LEN)
  // nnz: (NUM_Q_HEAD,)
  int64_t NUM_Q_HEAD = attn_weight.size(0);
  int64_t NUM_K_HEAD = value.size(0);
  int64_t SEQ_LEN = value.size(1);
  int64_t HEAD_DIM = value.size(2);
  int64_t GROUP = NUM_Q_HEAD / NUM_K_HEAD;
  batch_sparse_attention_wv_impl(
      static_cast<bfloat16 *>(value.data_ptr()),
      static_cast<float *>(attn_weight.data_ptr()),
      static_cast<bfloat16 *>(attn_output.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      NUM_Q_HEAD, GROUP, SEQ_LEN, HEAD_DIM);

}

void batch_sparse_attention_fp32_impl(
const float *query, 
const bfloat16 *key,
float *attn_weight, 
int32_t *ind, 
int32_t *nnz,
const int64_t NUM_Q_HEAD, 
const int64_t GROUP, 
const int64_t SEQ_LEN, 
const int64_t HEAD_DIM) {
      
//  key bfloat16 (NUM_K_HEAD, SEQ_LEN, HEAD_DIM)
//  query float (NUM_Q_HEAD, SEQ_LEN, HEAD_DIM)
//  attn_weight float (NUM_Q_HEAD, SEQ_LEN)
//  ind int (NUM_Q_HEAD, SEQ_LEN)
//  nnz int (NUM_Q_HEAD,)
//  NUM_Q_HEAD = NUM_K_HEAD * GROUP
      #pragma omp parallel for schedule(static,1) num_threads(96)
      for (int i = 0; i < NUM_Q_HEAD; ++i){
        sparse_attention_qk_fp32_impl(
          key + (i/GROUP) * SEQ_LEN * HEAD_DIM,
          ind + i * SEQ_LEN,
          query + i * HEAD_DIM,
          attn_weight + i * SEQ_LEN,
          SEQ_LEN,
          HEAD_DIM,
          nnz[i]
        );
      }
}
void batch_sparse_attention(
  torch::Tensor query, 
  torch::Tensor key,
  torch::Tensor indices,
  torch::Tensor attn_weight, 
  torch::Tensor nnz) {
  
  int64_t NUM_Q_HEAD = query.size(0);
  int64_t NUM_K_HEAD = key.size(0);
  int64_t SEQ_LEN = key.size(1);
  int64_t HEAD_DIM = key.size(2);
  int64_t GROUP = NUM_Q_HEAD / NUM_K_HEAD;
  auto q_f32 = query.to(torch::kFloat32);
  batch_sparse_attention_fp32_impl(
      static_cast<float *>(q_f32.data_ptr()),
      static_cast<bfloat16 *>(key.data_ptr()),
      static_cast<float *>(attn_weight.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      NUM_Q_HEAD, GROUP, SEQ_LEN, HEAD_DIM);
}
// TODO(Zihao): implement this function
void gather_gemv_outer_product_bf16_impl(const bfloat16 *W, const int32_t *ind,
                                         const float *x, bfloat16 *y,
                                         const int64_t M, const int64_t N,
                                         const int64_t nnz) {
    float y_f32[16];
    for (uint32_t io = 0; io < (N / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);

        for(uint32_t jo = 0; jo < nnz; ++jo)
          {
            const uint32_t row = ind[jo];
            __m512 x_j = _mm512_set1_ps(x[jo]);
            float w_f32[16];
            Bfloat16ToFloat_avx512(W + row * N + io * 16, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, y + io * 16, 16);
    }
                                         }


void gather_gemv_outer_product_bf16_128_impl(const bfloat16 *W, const int32_t *ind,
                                         const float *x, bfloat16 *y,
                                         const int64_t M, const int64_t N,
                                         const int64_t nnz) {
    float y_f32[128];
    __m512 y_acc[8];
    int64_t nnz_16 = (nnz % 16 == 0) ? (nnz): (nnz + 16 - nnz % 16);
    for(int i = 0; i < 128; ++i){
        y_f32[i] = 0.f;
    }
    for(int i = 0; i < 8; ++i){
      y_acc[i] = _mm512_loadu_ps(y_f32 + 16 * i);
    }
    for(uint32_t jo = 0; jo < nnz_16; ++jo)
    { 
      const uint32_t row = ind[jo];
      __m512 x_j = _mm512_set1_ps(x[jo]);
      for(int k = 0; k < 8; ++k){

          float w_f32[16];
          Bfloat16ToFloat_avx512(W + row * N + k * 16, w_f32, 16);
          __m512 w_tile = _mm512_loadu_ps(w_f32);
          y_acc[k] = _mm512_fmadd_ps(w_tile, x_j, y_acc[k]);
      }        
    
    }
    for (int i = 0; i < 8; ++i){
      _mm512_storeu_ps(y_f32 + 16 * i, y_acc[i]);
    }

    FloatToBfloat16_avx512(y_f32, y, 128);

                                         }


void batched_gather_gemv_outer_product_bf16_128_impl(const bfloat16 *W,
                                         const float *x, bfloat16 *y, int32_t *ind, int32_t *nnz,
                                         const int64_t BSZ, const int64_t M, const int64_t N, const int64_t ind_offset, const int64_t output_offset) {
      
      #pragma omp parallel for schedule(static,1) num_threads(96)
      for (int i = 0; i < BSZ; ++i){
        gather_gemv_outer_product_bf16_impl(
            W + i * M * N,
            ind + i * ind_offset,
            x + i * ind_offset,
            y + i * output_offset,
            M,
            N,
            nnz[i]
        );
      }
}
torch::Tensor gather_gemv_inner_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x) {
  // W: (M, N)
  // x: (N)
  // y: (nnz,)
  int64_t N = x.size(0);
  int64_t M = W.size(0);
  int64_t nnz = indices.size(0);
  auto y = torch::empty(
      {
          nnz,
      },
      x.options());
  auto x_f32 = x.to(torch::kFloat32);

  gather_gemv_inner_product_bf16_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<float *>(x_f32.data_ptr()),
      static_cast<bfloat16 *>(y.data_ptr()), M, N, nnz);

  return y;
}

void batch_gather_gemv_inner_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x, torch::Tensor y, torch::Tensor nnz) {
  // W: (BSZ, M, N)
  // x: (BSZ, N)
  // y: (BSZ, M)
  // indices: (BSZ, ind_offset)
  // nnz: (BSZ,)
  int64_t BSZ = W.size(0);
  int64_t M = W.size(1);
  int64_t N = W.size(2);
  int64_t ind_offset = indices.size(1);
  auto x_f32 = x.to(torch::kFloat32);
  batched_gather_gemv_inner_product_bf16_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<float *>(x_f32.data_ptr()),
      static_cast<bfloat16 *>(y.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      BSZ, M, N, ind_offset);

}
void batch_gather_gemv_inner_fp32_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x, torch::Tensor y, torch::Tensor nnz) {
  // W: (BSZ, M, N)
  // x: (BSZ, N)
  // y: (BSZ, M)
  // indices: (BSZ, ind_offset)
  // nnz: (BSZ,)
  int64_t BSZ = W.size(0);
  int64_t M = W.size(1);
  int64_t N = W.size(2);
  int64_t ind_offset = indices.size(1);
  auto x_f32 = x.to(torch::kFloat32);
  batched_gather_gemv_inner_product_fp32_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<float *>(x_f32.data_ptr()),
      static_cast<float *>(y.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      BSZ, M, N, ind_offset);
}
torch::Tensor gather_gemv_outer_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x) {
  // W: (M, N)
  // x: (nnz,)
  // y: (N,)
  // indices: (nnz,)
  int64_t M = W.size(0);
  int64_t N = W.size(1);
  int64_t nnz = indices.size(0);
  auto y = torch::empty(
      {
          N,
      },
      x.options());
  auto x_f32 = x.to(torch::kFloat32);

  gather_gemv_outer_product_bf16_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<float *>(x_f32.data_ptr()),
      static_cast<bfloat16 *>(y.data_ptr()), M, N, nnz);

  return y;
}

void batch_gather_gemv_outer_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x, torch::Tensor y, torch::Tensor nnz) {
  // W: (BSZ, M, N)
  // x: (BSZ, M)
  // y: (BSZ, N)
  // indices: (BSZ, ind_offset)
  // nnz: (BSZ,)
  int64_t BSZ = W.size(0);
  int64_t M = W.size(1);
  int64_t N = W.size(2);
  int64_t ind_offset = indices.size(1);
  int64_t output_offset = y.size(1);
  auto x_f32 = x.to(torch::kFloat32);
  batched_gather_gemv_outer_product_bf16_128_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<float *>(x_f32.data_ptr()),
      static_cast<bfloat16 *>(y.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      BSZ, M, N, ind_offset, output_offset);

}


void batch_gather_gemv_outer_fp32_product(torch::Tensor W, torch::Tensor indices,
                                        torch::Tensor x, torch::Tensor y, torch::Tensor nnz) {
  // W: (BSZ, M, N)
  // x: (BSZ, M)
  // y: (BSZ, N)
  // indices: (BSZ, ind_offset)
  // nnz: (BSZ,)
  int64_t BSZ = W.size(0);
  int64_t M = W.size(1);
  int64_t N = W.size(2);
  int64_t ind_offset = indices.size(1);
  int64_t output_offset = y.size(1);
  batched_gather_gemv_outer_product_bf16_128_impl(
      static_cast<bfloat16 *>(W.data_ptr()),
      static_cast<float *>(x.data_ptr()),
      static_cast<bfloat16 *>(y.data_ptr()),
      static_cast<int32_t *>(indices.data_ptr()),
      static_cast<int32_t *>(nnz.data_ptr()),
      BSZ, M, N, ind_offset, output_offset);

}

float vector_softmax(torch::Tensor x, int64_t nnz) {

  int64_t length = x.size(0);
  return softmax_fp32_impl(
    static_cast<float *>(x.data_ptr()),
    length,
    nnz
  );
}

void batch_softmax(torch::Tensor x, torch::Tensor nnz, torch::Tensor max_value_expsum) {

  int64_t bsz = x.size(0);
  int64_t length = x.size(1);

  batch_softmax_fp32_impl(
    static_cast<float *>(x.data_ptr()),
    static_cast<int *>(nnz.data_ptr()),
    static_cast<float *>(max_value_expsum.data_ptr()),
    length,
    bsz
  );
}

void batch_transform(torch::Tensor x, torch::Tensor nnz, torch::Tensor q_norm, torch::Tensor k_norm, int k, int l, float sqrt_dim, torch::Tensor indices) {

  int64_t bsz = x.size(0);
  int64_t length = x.size(1);

  batch_transform_fp32_impl(
    static_cast<float *>(x.data_ptr()),
    static_cast<int *>(nnz.data_ptr()),
    static_cast<float *>(q_norm.data_ptr()),
    static_cast<float *>(k_norm.data_ptr()),
    k,
    l,
    sqrt_dim,
    length,
    bsz,
    static_cast<int *>(indices.data_ptr())
  );
}


PYBIND11_MODULE(gather_gemv_cpu, m) {
  m.def("gather_gemv_inner_product", &gather_gemv_inner_product,
        "Computer y = W[ind] @ x, using inner product");
  
  m.def("batch_gather_gemv_inner_product", &batch_gather_gemv_inner_product,
        "Computer y = W[ind] @ x, using inner product");

  m.def("batch_gather_gemv_inner_fp32_product", &batch_gather_gemv_inner_fp32_product,
        "Computer y = W[ind] @ x, using inner product");
  
  
  m.def("gather_gemv_outer_product", &gather_gemv_outer_product,
        "Computer y = x @ W[ind], using outer product");
  
  m.def("batch_gather_gemv_outer_product", &batch_gather_gemv_outer_product,
        "Computer y = W[ind] @ x, using outer product");
  
  m.def("batch_gather_gemv_outer_fp32_product", &batch_gather_gemv_outer_fp32_product,
        "Computer y = W[ind] @ x, using outer product");
  
  m.def("vector_softmax", &vector_softmax,
        "Computer x = softmax(x), return max(x)");

  m.def("batch_softmax", &batch_softmax,
        "Computer x = softmax(x)");
  
  m.def("batch_transform", &batch_transform,
        "Computer x = transform(x) for lsh estimation");
  
  m.def("batch_sparse_attention", &batch_sparse_attention,
        "batch_sparse_attention");
  m.def("batch_sparse_attention_wv", &batch_sparse_attention_wv,
        "batch_sparse_attention_wv");
}

