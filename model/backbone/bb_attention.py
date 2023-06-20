
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from torch import Tensor
from typing import Optional
from torch import nn
import numpy as np
from .modeling_outputs import BaseAttentionOutput
from einops import rearrange, repeat
from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from operator import mul
from functools import partial, reduce, wraps
from itertools import chain
from torch.nn import Identity

from .reversible import ReversibleSequence

class SlefAttetion(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.mlp_attention =  nn.Linear(config.dim,config.dim*3)
        self.projection_out = nn.Linear(config.dim,config.dim)
        self.scale = 1/(np.sqrt())
    
    def forward(self,x,attn_mask):
        batch,sequence,dim = x.shape
        device= x.device
        qkv = self.mlp_attention(x)
        q,k,v = torch.chunk(qkv,chunks=3,dim=-1)










class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection  attention 层面果然还有一个线性层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.use_SDPA)
        self.visualize  = config.visualize
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att_after_softmax = F.softmax(att, dim=-1)
            att = self.attn_dropout(att_after_softmax)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if self.visualize:
            return BaseAttentionOutput(
                x = y,
                attention = att_after_softmax,
                queries=q,
                keys=k,
                values=v
            )
        else:
            return BaseAttentionOutput(x = y)



class BidirectionalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection  attention 层面果然还有一个线性层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.use_SDPA)

    def forward(self, x, attn_mask=None):
        if attn_mask is None:
            attn_mask = torch.ones(x.size(1), x.size(1)).to(x.device)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout_rate=0.0,
        causal=False,
        has_outproj=True,
        act_fun="relu",
        batch_first=True, # if True, input shape is (batch, seq_len, embed_dim)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # outprojection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # dropout rate
        self.dropout_rate = dropout_rate
        # causal
        self.causal = causal
        # batch first
        self.batch_first = batch_first

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation,compute Q',and K'
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform, 最大的长度系数
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d) compute Qcos and Qsin
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v) # 这里就是元素乘法，而不是矩阵乘法，
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)  # 这个地方相当于直接在time_length上累积
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)  求分母
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d) # 这里是合理的，对于每个v的dimension都要除以这么一个标量
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output

    def left_product(  # 这里的左乘仅仅是为了验证答案正确性
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output


class DirectMultiplyAttentionNotRight(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection  attention 层面果然还有一个线性层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout


    def forward(self, x,attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        

        # kv matrix multiply
        # (B, nh, T, d) (B, nh, d, T) -> (B, nh, d, d)
        mb = torch.matmul(k.transpose(2,3), v)
        mb = mb / math.sqrt(v.size(-1))
        # q matrix multiply
        # (B, nh, T, d) (B, nh, d, d) -> (B, nh, T, d)
        y = torch.matmul(q, mb)
        # (B, nh, T, d) -> (B, T, nh, d) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return BaseAttentionOutput(x = y)

from functools import partial, reduce, wraps
TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper fns
# copy from https://github.com/lucidrains/reformer-pytorch

def exists(val):
    return val is not None

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def default(val, default_val):
    return default_val if val is None else val

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cache_fn(f):  # 这个函数只执行一次，缓存cache
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def cache_method_decorator(cache_attr, cache_namespace, reexecute = False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose = False, normalize = False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x):
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

# copy from https://github.com/lucidrains/reformer-pytorch
# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

class LSHAttention(nn.Module):
    def __init__( self,
                  usr_lsh_hash = False,
                  usr_repeat_time:int = 5,
                  usr_topk:int = 100,
                  usr_dim_multiplier : int = 1,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)
        self.usr_lsh_hash = usr_lsh_hash

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

        self.usr_repeat_time = usr_repeat_time
        self.usr_topk = usr_topk
        self.usr_dim_multiplier = usr_dim_multiplier
    
    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def keep_largest_k_speed(self,tensor, k):
        # 计算张量中的绝对值
        abs_tensor = torch.abs(tensor)
        batch,sequence,dim = abs_tensor.shape
        # 找到绝对值最小的K个值的 索引
        _, idx = torch.topk(-1 * abs_tensor.view(-1, dim), dim-k)  # idx[batch_size*sequence_length, k ]
        # 将除了最大的K个值之外的所有值变为零
        tensor = tensor.view(-1, dim).scatter_(1, idx, 0) # 1维， index索引， 0值
        tensor = tensor.view(batch, sequence, dim)
        
        return tensor

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors_usr(self, x, m:int=1, T:int=5, topk:int = 100):
        device = x.device 
        batch_size, sequence_length, dim = x.shape
        y = torch.zeros(batch_size, sequence_length, m*dim).to(device=device)
        
        Z_shape = (m,dim, dim)
        Z = torch.normal(0, 1, size=Z_shape) * np.sqrt(dim) / np.sqrt(m*dim)
        Rs = []
        for i in range(m):
            U, S, Vh = torch.linalg.svd(Z[i], full_matrices=False)
            V = Vh.mH
            R = U @ V.transpose(1, 0)
            Rs.append(R)
        W = torch.stack(Rs, dim=1).to(device=device)  # W.shape = ( dim, m, dim)
        W = W.reshape(dim, m*dim).contiguous()
        W_tW = torch.matmul(W.transpose(1, 0), W)
        I = torch.eye(m*dim).to(device=device)  # 不同的batch用的相同的R
        I_sub_WtW = I - W_tW    
        alpha = np.sqrt(m*dim)/(2 * torch.mean(torch.norm(x, dim=-1))) # 这个之后可能需要动量

        for t in range(1, T+1): # 这个不带transpose 居然也可以过
            y_t = self.keep_largest_k_speed(alpha * torch.matmul(x,W) + 
                torch.matmul(y,I_sub_WtW),k = topk)
            y = y_t
                    
        buckets = torch.argmax(y, dim=-1)
        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        n_buckets = m * dim
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)
        # 下面这步就是复制R矩阵，一般_random_rotations_per_head是false
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, pos_emb = None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen) #这里已近把头给处理好了
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)
        if self.usr_lsh_hash:
            buckets = self.hash_vectors_usr(x = qk, m = self.usr_dim_multiplier,T=self.usr_repeat_time, topk=self.usr_topk)
        else:
            buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training) # buckets size [batch size, seq_len*buckets_num]

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach() # [bacth*num_head,total_hashes * seqlen]

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        if exists(pos_emb):
            qk = apply_rotary_pos_emb(qk, pos_emb)

        st = (sticker % seqlen) # 应该是帮助沿着hash函数复制的
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots) # 这个减法是真没懂
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,)) # 根据dots_logsumexp去进行hash重平衡

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets


# simple full attention

class FullQKAttention(nn.Module):
    def __init__(self, causal = False, dropout = 0.):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0)

# Shared qk attention, using either full or LSH attention

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, usr_lsh_hash = False, usr_repeat_time:int = 5,usr_topk:int = 100,usr_dim_multiplier : int = 1, 
            heads = 8, bucket_size = 64, n_hashes = 8, causal = False, dim_head = None, attn_chunks = 1, 
            random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, num_mem_kv = 0, 
            one_value_head = False, use_full_attn = False, full_attn_thres = None, return_attn = False, post_attn_dropout = 0., 
            dropout = 0., n_local_attn_heads = 0, **kwargs):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias = False)  # 这个地方确实qk是一样的,dim_heads=64 * 8
        self.tov = nn.Linear(dim, v_dim, bias = False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(usr_lsh_hash=usr_lsh_hash, usr_repeat_time= usr_repeat_time,
                  usr_topk = usr_topk,
                  usr_dim_multiplier = usr_dim_multiplier,bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, dropout = dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size) # 这里是64就会

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True, look_forward=(1 if not causal else 0))

        self.callback = None

    def forward(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None, pos_emb = None, **kwargs):
        device, dtype = x.device, x.dtype
        b, t, e, h, dh, m, l_h = *x.shape, self.heads, self.dim_head, self.num_mem_kv, self.n_local_attn_heads

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, -1)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)  # 这里是qk最开始的地方
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))#在这里很明确的把qk的头给他merge进去了

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v)) 
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))  # lqk一般是空，啥都没有

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn # 主要是这里进行耦合，选择传统的attention还是LSHAttention
        partial_attn_fn = partial(attn_fn, query_len = t, pos_emb = pos_emb, **kwargs) # 参数固定
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks = self.attn_chunks)

        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out)

# feed forward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# reformer lm

class Reformer(nn.Module):
    def __init__(self, dim, depth,usr_repeat_time:int = 5,usr_topk:int = 100,usr_dim_multiplier : int = 1, usr_lsh_hash = False, 
    heads = 8, dim_head = None, bucket_size = 64, n_hashes = 8, ff_chunks = 100, 
    attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_activation = None, 
    ff_mult = 4, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., lsh_attend_across_buckets = True, 
    lsh_allow_duplicate_attention = True, random_rotations_per_head = False, use_scale_norm = False, use_rezero = False, 
    use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, 
    n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv

        self.full_attn_thres = full_attn_thres

        get_attn = lambda: LSHSelfAttention(dim,usr_lsh_hash,usr_repeat_time, usr_topk, usr_dim_multiplier, heads, bucket_size, n_hashes, causal = causal, dim_head = dim_head, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, activation = ff_activation, mult = ff_mult, glu = ff_glu), along_dim = -2)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        if weight_tie:
            get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_pkm))

        blocks = []

        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)
            parallel_net = None

            attn = get_attn()

            if use_pkm:
                parallel_net = get_pkm()
            else:
                parallel_net = get_ff()

            f = residual_fn_wrapper(attn)
            g = residual_fn_wrapper(parallel_net)

            blocks.append(nn.ModuleList([f, g]))

        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout = layer_dropout, reverse_thres = reverse_thres, send_signal = True)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        x = self.layers(x, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class LSHTreeAttention(nn.Module):
    def __init__( self,
                  usr_lsh_hash = False,
                  usr_repeat_time:int = 5,
                  usr_topk:int = 100,
                  usr_dim_multiplier : int = 1,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)
        self.usr_lsh_hash = usr_lsh_hash

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

        self.usr_repeat_time = usr_repeat_time
        self.usr_topk = usr_topk
        self.usr_dim_multiplier = usr_dim_multiplier
    

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)
        # 下面这步就是复制R矩阵，一般_random_rotations_per_head是false
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, pos_emb = None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen) #这里已近把头给处理好了
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)
        if self.usr_lsh_hash:
            buckets = self.hash_vectors_usr(x = qk, m = self.usr_dim_multiplier,T=self.usr_repeat_time, topk=self.usr_topk)
        else:
            buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training) # buckets size [batch size, seq_len*buckets_num]

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach() # [bacth*num_head,total_hashes * seqlen]

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        if exists(pos_emb):
            qk = apply_rotary_pos_emb(qk, pos_emb)

        st = (sticker % seqlen) # 应该是帮助沿着hash函数复制的
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots) # 这个减法是真没懂
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,)) # 根据dots_logsumexp去进行hash重平衡

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets

