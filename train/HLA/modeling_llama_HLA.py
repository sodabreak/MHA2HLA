# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig

# if is_torch_flex_attn_available():
#     from torch.nn.attention.flex_attention import BlockMask
#     from transformers.integrations.flex_attention import make_flex_block_causal_mask



logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaRotaryEmbeddingmy(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        inv_freq = inv_freq[: inv_freq.shape[0] // 2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            inv_freq = inv_freq[: inv_freq.shape[0] // 2]
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # 计算 RoPE 角度
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            # emb = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # 形状: [batch_size, seq_len, head_dim//2]
            sin = freqs.sin()  # 形状: [batch_size, seq_len, head_dim//2]

        # 应用 RoPE 位置缩放
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # 变换为旋转矩阵形式 `[seq_len, head_dim//2, 2, 2]`
        rope_matrix = torch.zeros((cos.shape[1], cos.shape[2], 2, 2), device=cos.device, dtype=cos.dtype)
        rope_matrix[..., 0, 0] = cos.squeeze(0)  # cos(θ)
        rope_matrix[..., 0, 1] = -sin.squeeze(0)  # -sin(θ)
        rope_matrix[..., 1, 0] = sin.squeeze(0)  # sin(θ)
        rope_matrix[..., 1, 1] = cos.squeeze(0)  # cos(θ)

        return rope_matrix  # [seq_len, head_dim//2, 2, 2]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_hla(q, k, cos_size_matrix, B_q, B_k, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors with head-level adaptation.

    Args:
        q (`torch.Tensor`): The query tensor. Shape: [batch, num_heads, seq_len, 32]
        k (`torch.Tensor`): The key tensor. Shape: [batch, num_heads, seq_len, 32]
        cos_size_matrix (`torch.Tensor`): The RoPE matrix. Shape: [seq_len, head_dim//2, 2, 2]
        B_q (`torch.Tensor`): The transformation matrix for q. Shape: [num_heads*32, num_heads*32]
        B_k (`torch.Tensor`): The transformation matrix for k. Shape: [num_heads*32, num_heads*32]
        position_ids (`torch.Tensor`, *optional*): Position indices (not used here)
        unsqueeze_dim (`int`, *optional*, defaults to 1): 
            Specifies the dimension along which to unsqueeze for broadcasting.
    
    Returns:
        `tuple(torch.Tensor)`: The transformed query and key tensors.
    """
    batch_size, num_heads_q, seq_len, head_dim = q.shape  # [batch, num_heads, seq_len, 32]
    batch_size, num_heads_k, seq_len, head_dim = k.shape  
    head_dim_half = head_dim // 2  # 32 -> 16
    total_dim_q = num_heads_q * head_dim  # `num_heads * 32`
    total_dim_k = num_heads_k * head_dim

    q_embed = torch.zeros_like(q)
    k_embed = torch.zeros_like(k)

    q_concat = q.permute(0, 2, 1, 3).reshape(batch_size, seq_len, total_dim_q)  # [batch, seq_len, num_heads*32]
    k_concat = k.permute(0, 2, 1, 3).reshape(batch_size, seq_len, total_dim_k)  # [batch, seq_len, num_heads*32]

    q_transformed = torch.zeros_like(q_concat)  # [batch, seq_len, num_heads*32]
    k_transformed = torch.zeros_like(k_concat)  # [batch, seq_len, num_heads*32]

    for i in range(head_dim_half * num_heads_q):
        rope_matrix_h = cos_size_matrix[:, i%head_dim_half, :, :]  # [seq_len, 2, 2]
        B_q_2 = B_q[2*i:2*(i+1), :]  # [2, num_heads*32]
        # `B' * R`
        B_q_rot = torch.matmul(B_q_2.T.to(torch.float16), rope_matrix_h.to(torch.float16))  # Convert both to float16
        # `(B' R) B`
        B_q_transformed = torch.matmul(B_q_rot, B_q_2)  # [seq_len, num_heads*32, num_heads*32]
        q_h = q_concat  # [batch, seq_len, num_heads*32]
        # `q' = (B' R B) q`
        q_final = torch.einsum("bsi,sij->bsj", q_h, B_q_transformed)  # [batch, seq_len, num_heads*32]
        q_transformed += q_final

    for i in range(head_dim_half * num_heads_k):
        rope_matrix_h = cos_size_matrix[:, i%head_dim_half, :, :]  # [seq_len, 2, 2]
        B_k_2 = B_k[2*i:2*(i+1), :]  # [2, num_heads*32]
        # `B' * R`
        B_k_rot = torch.matmul(B_k_2.T.to(torch.float16), rope_matrix_h.to(torch.float16))  # 
        # `(B' R) B`
        B_k_transformed = torch.matmul(B_k_rot, B_k_2)  # [seq_len, num_heads*32, num_heads*32]
        k_h = k_concat  # [batch, seq_len, num_heads*32]
        # `q' = (B' R B) q`
        k_final = torch.einsum("bsi,sij->bsj", k_h, B_k_transformed)  # [batch, seq_len, num_heads*32]
        k_transformed += k_final

    q_embed = q_transformed.view(batch_size, seq_len, num_heads_q * head_dim)
    k_embed = k_transformed.view(batch_size, seq_len, num_heads_k * head_dim)

    return q_embed, k_embed

def apply_rotary_pos_emb_hla_fast(q, k, cos_size_matrix, B_q, B_k):
    batch_size, num_heads_q, seq_len, head_dim = q.shape
    _, num_heads_k, _, _ = k.shape
    head_dim_half = head_dim // 2
    total_dim_q = num_heads_q * head_dim
    total_dim_k = num_heads_k * head_dim

    # Merge multi-head dimensions [batch, seq_len, total_dim]
    q_concat = q.permute(0, 2, 1, 3).reshape(batch_size, seq_len, total_dim_q)
    k_concat = k.permute(0, 2, 1, 3).reshape(batch_size, seq_len, total_dim_k)

    # Core transformation function
    def parallel_transform(x, B, num_heads, cos_matrix):
        # Parameter reorganization
        num_blocks = num_heads * head_dim_half  # Total blocks = num_heads × 16
        # Reorganize B matrix to [seq_len, num_blocks, 2, total_dim]
        B_blocks = B.view(num_blocks, 2, -1)  # [n_blocks, 2, D]
        B_blocks = B_blocks.unsqueeze(0).expand(seq_len, -1, -1, -1) # [s, n_blocks, 2, D]
        
        # Expand cos matrix to [seq_len, num_blocks, 2, 2]
        cos_expanded = cos_matrix[:, :head_dim_half]  # [s, 16, 2, 2]
        cos_expanded = cos_expanded.unsqueeze(1)  # [s, 1, 16, 2, 2]
        cos_expanded = cos_expanded.expand(-1, num_heads, -1, -1, -1)  # [s, nh, 16, 2, 2]
        cos_expanded = cos_expanded.reshape(seq_len, num_blocks, 2, 2)  # [s, n_blocks, 2, 2]

        # Calculate B'R [s, n_blocks, D, 2]
        B_rot = torch.einsum('snij,snjk->snik', 
                            B_blocks.transpose(2,3),  # [s,n_blocks,D,2]
                            cos_expanded.to(torch.float16) )              # [s,n_blocks,2,2]
        
        # Calculate (B'R)B [s, n_blocks, D, D]
        B_trans = torch.einsum('snik,snkj->snij',  # Key dimension alignment fix
                             B_rot,                # [s,n_blocks,D,2]
                             B_blocks)  # [s,n_blocks,2,D]
        
        # Apply transformation and accumulate [batch, seq_len, D]
        x_trans = torch.einsum('bsd,snij->bsnj', 
                              x,  # [batch, s, D]
                              B_trans)              # [s,n_blocks,D,D]
        return x_trans.sum(dim=2).to(x.dtype)       # Sum along block dimension

    # Execute transformations
    q_transformed = parallel_transform(q_concat, B_q, num_heads_q, cos_size_matrix)
    k_transformed = parallel_transform(k_concat, B_k, num_heads_k, cos_size_matrix)

    # Restore original shape [batch, num_heads, seq_len, head_dim]
    q_embed = q_transformed.view(batch_size, seq_len, num_heads_q * head_dim)
    k_embed = k_transformed.view(batch_size, seq_len, num_heads_k * head_dim)

    return q_embed, k_embed

def apply_rotary_pos_emb_hla_fast_opt(q, k, cos_size_matrix, B_q, B_k):
    batch_size, num_heads_q, seq_len, head_dim = q.shape
    _, num_heads_k, _, _ = k.shape
    head_dim_half = head_dim // 2
    total_dim_q = num_heads_q * head_dim
    total_dim_k = num_heads_k * head_dim

    # 合并多头维度并确保内存连续
    q_concat = q.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, total_dim_q)
    k_concat = k.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, total_dim_k)

    def parallel_transform(x, B, num_heads, cos_matrix):
        device = x.device
        dtype = x.dtype
        num_blocks = num_heads * head_dim_half
        s = seq_len
        D = B.shape[-1]

        # 处理B矩阵
        B_blocks = B.view(num_blocks, 2, D)
        B_blocks = B_blocks.unsqueeze(0).expand(s, -1, -1, -1)  # [s, n, 2, D]
        s, n, two, D = B_blocks.shape
        B_blocks = B_blocks.contiguous().view(s * n, two, D)

        # 处理cos矩阵
        cos_expanded = cos_matrix[:, :head_dim_half].unsqueeze(1)  # [s, 1, 16, 2, 2]
        cos_expanded = cos_expanded.expand(-1, num_heads, -1, -1, -1).contiguous()  # [s, nh, 16, 2, 2]
        cos_expanded = cos_expanded.view(s, num_blocks, 2, 2).contiguous()  # [s, n, 2, 2]
        cos_expanded = cos_expanded.view(s * n, 2, 2).to(dtype).to(device)

        # 批量矩阵乘法计算B_rot和B_trans
        B_blocks_float = B_blocks.to(dtype)
        B_blocks_transposed = torch.transpose(B_blocks_float, 1, 2)  # [s*n, D, 2]
        B_rot = torch.bmm(B_blocks_transposed, cos_expanded)  # [s*n, D, 2]
        B_trans = torch.bmm(B_rot, B_blocks_float)  # [s*n, D, D]
        B_trans = B_trans.view(s, n, D, D)  # [s, n, D, D]

        # 应用变换
        x_float = x.to(dtype)
        x_trans = torch.einsum('bsd,sndd->bsnd', x_float, B_trans)  # 自动广播
        return x_trans.sum(dim=2).to(dtype)

    # 执行变换并恢复原始形状
    q_transformed = parallel_transform(q_concat, B_q, num_heads_q, cos_size_matrix)
    k_transformed = parallel_transform(k_concat, B_k, num_heads_k, cos_size_matrix)

    q_embed = q_transformed.view(batch_size, seq_len, num_heads_q* head_dim)
    k_embed = k_transformed.view(batch_size, seq_len, num_heads_k* head_dim)

    return q_embed, k_embed

def apply_rotary_pos_emb_hla_fast_opt_v2(q, k, cos_size_matrix, B_q, B_k):
    batch_size, num_heads_q, seq_len, head_dim = q.shape
    _, num_heads_k, _, _ = k.shape
    head_dim_half = head_dim // 2
    device = q.device
    dtype = q.dtype

    # ---------------------- 预计算静态参数 ----------------------
    # 将B矩阵预处理为 [s*n, 2, D] 形式 (避免运行时重复展开)
    def preprocess_B(B, num_heads):
        num_blocks = num_heads * head_dim_half
        B_blocks = B.view(num_blocks, 2, -1)  # [n, 2, D]
        return B_blocks.unsqueeze(0).expand(seq_len, -1, -1, -1).reshape(-1, 2, B.shape[-1])  # [s*n, 2, D]

    # 将cos矩阵预处理为 [s*n, 2, 2] 形式
    def preprocess_cos(cos, num_heads):
        return (cos[:, :head_dim_half]
                .unsqueeze(1)
                .expand(-1, num_heads, -1, -1, -1)
                .reshape(seq_len, num_heads * head_dim_half, 2, 2)
                .reshape(-1, 2, 2)
                .to(dtype=dtype, device=device))

    # ---------------------- 核心计算优化 ----------------------
    def optimized_transform(x, B, cos):
        # 合并矩阵乘法链: (B^T @ R) @ B → B^T @ (R @ B)
        B_float = B.to(dtype)
        R_B = torch.bmm(cos, B_float)  # [s*n, 2, D]
        BTRB = torch.bmm(B_float.transpose(1, 2), R_B)  # [s*n, D, D]
        
        # 重组为 [s, n, D, D]
        BTRB = BTRB.view(seq_len, -1, BTRB.shape[-2], BTRB.shape[-1])
        
        # 高效矩阵乘法替代 einsum
        x = x.view(batch_size, seq_len, 1, -1)  # [b, s, 1, nh*d]
        x_expanded = x.unsqueeze(3)  # [b, s, 1, 1, D]
        result = torch.matmul(x_expanded, BTRB.unsqueeze(0))  # [b, s, 1, 1, D]
        return result.squeeze(3).squeeze(2).sum(dim=2)  # [b, s, D]

    # ---------------------- 执行流程 ----------------------
    # 预处理参数
    B_q_pre = preprocess_B(B_q, num_heads_q)
    B_k_pre = preprocess_B(B_k, num_heads_k)
    cos_q = preprocess_cos(cos_size_matrix, num_heads_q)
    cos_k = preprocess_cos(cos_size_matrix, num_heads_k)

    # 合并多头维度
    q_concat = q.permute(0, 2, 1, 3).flatten(2)  # [b, s, nh*d]
    k_concat = k.permute(0, 2, 1, 3).flatten(2)

    # 执行变换
    q_trans = optimized_transform(q_concat, B_q_pre, cos_q)
    k_trans = optimized_transform(k_concat, B_k_pre, cos_k)

    # 恢复形状
    return q_trans.view(batch_size, seq_len, num_heads_q* head_dim), k_trans.view(batch_size, seq_len, num_heads_k* head_dim)

# # 将内部函数提取为静态方法（避免动态闭包影响编译）
# @torch.compile(dynamic=True, fullgraph=False, mode="max-autotune")
# def parallel_transform_compiled(
#     x: torch.Tensor, 
#     B_blocks: torch.Tensor, 
#     cos_expanded: torch.Tensor,
#     num_blocks: int,
#     D: int
# ) -> torch.Tensor:
#     # 批量矩阵计算优化为单一kernel
#     B_blocks_float = B_blocks.to(x.dtype)
    
#     # 转置与矩阵乘法融合
#     B_rot = torch.bmm(
#         B_blocks_float.transpose(1,2),  # [s*n, D, 2]
#         cos_expanded                     # [s*n, 2, 2]
#     )
    
#     # 矩阵链式乘法优化
#     B_trans = torch.bmm(B_rot, B_blocks_float)  # [s*n, D, D]
#     B_trans = B_trans.view(-1, num_blocks, D, D)  # [s, n, D, D]
    
#     # 替换einsum为广播乘法（编译更友好）
#     x_trans = (x.unsqueeze(2).to(x.dtype) @ B_trans.permute(0,1,3,2))
#     return x_trans.squeeze(2)

# def apply_rotary_pos_emb_compiled(q, k, cos_size_matrix, B_q, B_k):
#     # 静态形状提取（编译时固定）
#     batch_size, num_heads_q, seq_len, head_dim = q.shape
#     _, num_heads_k, _, _ = k.shape
#     head_dim_half = head_dim // 2
    
#     # 预计算B和cos的静态参数
#     B_q_preprocessed = _preprocess_B(B_q, seq_len, num_heads_q, head_dim_half)
#     B_k_preprocessed = _preprocess_B(B_k, seq_len, num_heads_k, head_dim_half)
#     cos_q = _preprocess_cos(cos_size_matrix, num_heads_q, head_dim_half)
#     cos_k = _preprocess_cos(cos_size_matrix, num_heads_k, head_dim_half)

#     # 合并多头维度（保持连续内存布局）
#     q_concat = q.permute(0,2,1,3).contiguous().view(batch_size, seq_len, -1)
#     k_concat = k.permute(0,2,1,3).contiguous().view(batch_size, seq_len, -1)

#     # 执行编译优化的核心变换
#     q_trans = parallel_transform_compiled(
#         q_concat, B_q_preprocessed, cos_q, 
#         num_heads_q * head_dim_half, B_q.shape[-1]
#     )
#     k_trans = parallel_transform_compiled(
#         k_concat, B_k_preprocessed, cos_k,
#         num_heads_k * head_dim_half, B_k.shape[-1]
#     )

#     # 输出形状重构
#     return q_trans.view(batch_size, seq_len, num_heads_q* head_dim), k_trans.view(batch_size, seq_len, num_heads_k* head_dim)

# # 预处理器函数（静态参数计算）
# def _preprocess_B(B: torch.Tensor, seq_len: int, num_heads: int, head_dim_half: int):
#     num_blocks = num_heads * head_dim_half
#     B_blocks = B.view(num_blocks, 2, -1)
#     B_expanded = B_blocks.unsqueeze(0).expand(seq_len, -1, -1, -1)  # [s, n, 2, D]
#     return B_expanded.contiguous().view(seq_len * num_blocks, 2, -1)

# def _preprocess_cos(cos: torch.Tensor, num_heads: int, head_dim_half: int):
#     cos_expanded = cos[:, :head_dim_half].unsqueeze(1)  # [s, 1, 16, 2, 2]
#     cos_expanded = cos_expanded.expand(-1, num_heads, -1, -1, -1).contiguous()
#     return cos_expanded.view(cos.size(0), num_heads * head_dim_half, 2, 2).view(-1, 2, 2)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.init = False
        self.init_B = False
        # self.B_q = nn.Parameter(torch.zeros(self.head_dim//2*config.num_attention_heads, self.head_dim//2*config.num_attention_heads))
        # self.B_k = nn.Parameter(torch.zeros(self.head_dim//2*config.num_key_value_heads, self.head_dim//2*config.num_key_value_heads))

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # new layers
        self.q_d_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim // 2
        )
        self.q_u_proj = nn.Linear(
            config.num_attention_heads * self.head_dim // 2, config.num_attention_heads * self.head_dim
        )
        self.k_d_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads  * self.head_dim // 2
        )
        self.k_u_proj = nn.Linear(
            config.num_key_value_heads * self.head_dim // 2, config.num_key_value_heads * self.head_dim
        )
        self.v_d_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim // 2
        )
        self.v_u_proj = nn.Linear(
            config.num_key_value_heads * self.head_dim // 2, config.num_key_value_heads * self.head_dim
        )


    def get_up_down_matrix(self):
        """
        对 q_proj、k_proj 分别用SVD分解，然后把分解后的矩阵拷贝到
         q_d_proj、q_u_proj，以及 k_d_proj、k_u_proj 上。
        """
        with torch.no_grad():
            W_q = self.q_proj.weight.float()  # [384, 384]
            U_q, S_q, Vh_q = torch.linalg.svd(W_q, full_matrices=False)  # U_q: [384,384], S_q: [384], Vh_q: [384,384]

            r_q = W_q.shape[0]//2
            U_qr = U_q[:, :r_q]                # [384,192]
            S_qr = torch.diag(S_q[:r_q])       # [192,192]
            Vh_qr = Vh_q[:r_q, :]              # [192,384]

            up_q = U_qr
            down_q = S_qr @ Vh_qr

            # 拷到 q_d_proj, q_u_proj
            self.q_d_proj.weight.copy_(down_q.to(self.q_proj.weight.dtype))  # q_d_proj.weight -> [192,384]
            self.q_u_proj.weight.copy_(up_q.to(self.q_proj.weight.dtype))    # q_u_proj.weight -> [384,192]

            W_k = self.k_proj.weight.float()  # [128, 384]
            U_k, S_k, Vh_k = torch.linalg.svd(W_k, full_matrices=False)  # U_k: [128,128], S_k: [128], Vh_k: [128,384]

            # 取 rank=64
            r_k = W_k.shape[0]//2
            U_kr = U_k[:, :r_k]               # [128,64]
            S_kr = torch.diag(S_k[:r_k])      # [64,64]
            Vh_kr = Vh_k[:r_k, :]             # [64,384]

            up_k = U_kr
            down_k = S_kr @ Vh_kr

            # 拷到 k_d_proj, k_u_proj
            self.k_d_proj.weight.copy_(down_k.to(self.q_proj.weight.dtype))  # k_d_proj.weight -> [64,384]
            self.k_u_proj.weight.copy_(up_k.to(self.q_proj.weight.dtype))    # k_u_proj.weight -> [128,64]

            W_v = self.v_proj.weight.float()  # [128, 384]
            U_v, S_v, Vh_v = torch.linalg.svd(W_v, full_matrices=False)  # U_k: [128,128], S_k: [128], Vh_k: [128,384]

            # 取 rank=64
            r_v = W_v.shape[0]//2
            U_vr = U_v[:, :r_v]               # [128,64]
            S_vr = torch.diag(S_v[:r_v])      # [64,64]
            Vh_vr = Vh_v[:r_v, :]             # [64,384]

            up_v = U_vr
            down_v = S_vr @ Vh_vr

            # 拷到 k_d_proj, k_u_proj
            self.v_d_proj.weight.copy_(down_v.to(self.q_proj.weight.dtype))  # k_d_proj.weight -> [64,384]
            self.v_u_proj.weight.copy_(up_v.to(self.q_proj.weight.dtype))  

    @staticmethod
    def randomized_svd(A, rank=None, n_iter=5):
        """
        使用随机 SVD 分解，提高计算速度：
        - rank: 目标低秩近似的维度（默认为 X.shape[1]）
        - n_iter: 迭代次数，控制近似精度
        """
        # if not isinstance(X, torch.Tensor):
        #     raise ValueError(f"Expected a tensor, but got {type(X)}")
        if rank is None:
            rank = min(A.shape)  
        Q_init = torch.randn(A.shape[0], rank, device=A.device, dtype=A.dtype)
        Q, _ = torch.linalg.qr(Q_init)  
        B = Q.T @ A  # 低秩投影
        U, _, V_T = torch.linalg.svd(B.float(), full_matrices=False)

        return V_T.T.to(A.dtype)  # 返回近似的 B 矩阵

    def get_up_cb_matrix_fast(self):
        """ 使用 Randomized SVD 替代标准 SVD，提高计算速度 """
        with torch.no_grad():
            W_q = self.q_u_proj.weight.float()
            B_q = LlamaAttention.randomized_svd(W_q)  # 近似 B 矩阵
            W_k = self.k_u_proj.weight.float()
            B_k = LlamaAttention.randomized_svd(W_k)

        return B_q.to(self.q_u_proj.weight.dtype), B_k.to(self.q_u_proj.weight.dtype)
    

    def get_up_cb_matrix(self):
        """ input: up_matrix: [32 * config.hidden_size, 64 * config.hidden_size] [32 * 6, 64 * 6]
            output: C(U): [64 * 6, 64 * 6] 
                    S(sigma): [64 * 6, 32 * 6]
                    B(V): [32 * 6, 32 * 6]
        """
        with torch.no_grad():
            W_q = self.q_u_proj.weight.float()
            C_q, S_q, B_q = torch.linalg.svd(W_q, full_matrices=False)
            W_k = self.k_u_proj.weight.float()
            C_k, S_k, B_k = torch.linalg.svd(W_k, full_matrices=False)

        return B_q.T.to(self.q_u_proj.weight.dtype), B_k.T.to(self.q_u_proj.weight.dtype), C_q.to(self.q_u_proj.weight.dtype), C_k.to(self.q_u_proj.weight.dtype)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        training: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if torch.all(self.k_u_proj.weight == 0) and training: 
        if not self.init:
            self.get_up_down_matrix()
            self.init=True

        input_shape = hidden_states.shape[:-1] # torch.Size([1, 5])
        hidden_shape = (*input_shape, -1, self.head_dim // 2) # (1, 5, -1, 64)

        query_states_h = self.q_d_proj(hidden_states).view(hidden_shape).transpose(1, 2) # torch.Size([1, 6, 5, 64])
        key_states_h = self.k_d_proj(hidden_states).view(hidden_shape).transpose(1, 2) # torch.Size([1, 2, 5, 64])
        value_states_h = self.v_d_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos_size_matrix = position_embeddings # (self.head_dim // 2, 2,2)

        B_q, B_k, C_q, C_k = self.get_up_cb_matrix()
        # if not self.init_B:
        #     B_q, B_k = self.get_up_cb_matrix_fast()
        #     self.B_q = nn.Parameter(B_q.clone().detach())
        #     self.B_k = nn.Parameter(B_k.clone().detach())
        #     self.init = True

        # query_states_h, key_states_h = apply_rotary_pos_emb_hla(query_states_h, key_states_h, cos_size_matrix, B_q, B_k)
        query_states_h, key_states_h = apply_rotary_pos_emb_hla_fast(query_states_h, key_states_h, cos_size_matrix,B_q,B_k)
        value_states_h = value_states_h.permute(0,2,1,3).view(*input_shape, -1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cos_sin_matrix":cos_size_matrix, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states_h, value_states_h, self.layer_idx, cache_kwargs)

        # query_states = self.q_u_proj(query_states_h).view(*input_shape, -1, self.head_dim).permute(0,2,1,3)
        # key_states = self.k_u_proj(key_states_h).view(*input_shape, -1, self.head_dim).permute(0,2,1,3)
        query_states = self.q_u_proj(query_states_h)
        key_states = self.k_u_proj(key_states_h)
        value_states = self.v_u_proj(value_states_h).view(*input_shape, -1, self.head_dim).permute(0,2,1,3)

        query_states_u = query_states@C_q
        key_state_u = key_states@C_k

        query_states[:,:,:query_states_u.shape[2]] = query_states_u
        key_states[:,:,:key_state_u.shape[2]] = key_state_u
        query_states = query_states.view(*input_shape, -1, self.head_dim).permute(0,2,1,3)
        key_states = key_states.view(*input_shape, -1, self.head_dim).permute(0,2,1,3)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        training: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states # torch.Size([1, 5, 384])

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            training=training,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length1.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.my_rotary_emb = LlamaRotaryEmbeddingmy(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        training: bool = True,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds #torch.Size([1, 5, 384])

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids) # (torch.Size([1, 5, 64]), torch.Size([1, 5, 64]))
        position_embeddings = self.my_rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    training=training,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size # 49152
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 384, 49152
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        training: bool = True,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            training=training,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# __all__ = [
#     "LlamaForCausalLM",
#     "LlamaModel",
#     "LlamaPreTrainedModel",
#     "LlamaForSequenceClassification",
#     "LlamaForQuestionAnswering",
#     "LlamaForTokenClassification",
# ]

if __name__ == '__main__':

    from transformers import  AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model name
    model_name = "AICrossSim/clm-60m"
    tokenizer_name = "HuggingFaceTB/cosmo2-tokenizer"

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    model = LlamaForCausalLM.from_pretrained(model_name,  ignore_mismatched_sizes=True, torch_dtype=torch.float16).to(device)

    input_text = "The future of AI is"
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_attention_mask=True
    ).to(device)
    
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask, 
        max_length=50, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        training=True
    ) # torch.Size([1, 50])

    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_text)