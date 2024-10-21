import torch
import gather_gemv_cpu
import numpy as np
import pytest
torch.set_printoptions(profile="full")

batch_sparse_attention = gather_gemv_cpu.batch_sparse_attention
@pytest.mark.parametrize("hq", [32, 64])
@pytest.mark.parametrize("group", [4, 8])
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("max_nnz", [256])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_sparse_attention(hq, group, seq_len, max_nnz, head_dim):

    
    key = torch.randn(hq // group, seq_len, head_dim).bfloat16()
    query = torch.ones(hq, head_dim).bfloat16()

    ind = torch.zeros((hq, seq_len)).int()
    
    nnz = torch.randint(low=0, high=max_nnz // 16, size=(hq,)).int()
    nnz = nnz * 16
    
    
    for i in range(hq):
        ind[i][:nnz[i]] = torch.randperm(seq_len)[:nnz[i]].int()
    
    
    attn_weight = torch.zeros((hq, seq_len)).float()

    batch_sparse_attention(query, key, ind, attn_weight, nnz)
    
    attn_weight_ref = torch.zeros(hq, seq_len).float()
    for i in range(hq):
        for j in range(nnz[i]):
            k = key[i // group][ind[i][j]]
            attn_weight_ref[i][j] = torch.dot(k, query[i])
    
    
    np.testing.assert_allclose(attn_weight.float().numpy(), attn_weight_ref.float().numpy(), rtol=1e-2, atol=1e-3)

batch_wv = gather_gemv_cpu.batch_sparse_attention_wv
@pytest.mark.parametrize("hq", [32, 64])
@pytest.mark.parametrize("group", [4, 8])
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize("max_nnz", [1024])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_wv(hq, group, seq_len, max_nnz, head_dim):
    value = torch.randn(hq//group, seq_len, head_dim).bfloat16()
    attn_weight = torch.ones(hq, seq_len).float()
    
    ind = torch.zeros((hq, seq_len)).int()

    nnz = torch.randint(low=0, high=max_nnz // 16, size=(hq,)).int()
    nnz = nnz * 16
    
    for i in range(hq):
        ind[i][:nnz[i]] = torch.randperm(seq_len)[:nnz[i]].int()
    
    attn_output = torch.zeros((hq, head_dim)).bfloat16()
    
    batch_wv(value, ind, attn_weight, attn_output, nnz)

    y_ref = torch.zeros(hq, head_dim).bfloat16()
    for i in range(hq):
        x_i = attn_weight[i][:nnz[i]]
        w_i = value[i//group][ind[i][:nnz[i]]]
        
        y_ref[i] = torch.matmul(x_i.bfloat16(), w_i)
    
    
    assert torch.allclose(y_ref, attn_output, rtol=1e-2, atol=1e-2)

batch_softmax = gather_gemv_cpu.batch_softmax
@pytest.mark.parametrize("hq", [8, 16])
@pytest.mark.parametrize("seq_len", [192, 256, 384, 512])
@pytest.mark.parametrize("max_nnz", [128])
def test_batch_softmax(hq:int, seq_len:int, max_nnz:int):

    x = torch.randn(size=(hq, seq_len)).float()
    x_ref = x.clone()

    max_value_expsum = torch.zeros(size=(2, hq)).float()

    nnz = torch.randint(low=0, high=max_nnz // 16, size=(hq,)).int()
    nnz = nnz * 16

    batch_softmax(x, nnz, max_value_expsum)

    max_value = max_value_expsum[0]
    for i in range(hq):

        x_i = x_ref[i]
        nnz_i = nnz[i]
        if nnz_i > 0:
            x_i[nnz_i:] = -torch.inf
            m_i = torch.max(x_i)

            assert torch.abs(max_value[i] - m_i) < 1e-3

            x_i = torch.softmax(x_i[:nnz_i], dim=0)
            assert torch.allclose(x[i][:nnz_i], x_i, rtol=1e-2, atol=1e-3)
