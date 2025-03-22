import torch
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
        B_q_2 = B_q[2*i:2*(i+1), :].to(torch.float16)  # [2, num_heads*32]
        # `B' * R`
        B_q_rot = torch.matmul(B_q_2.T.to(torch.float16), rope_matrix_h.to(torch.float16))  # Convert both to float16
        # `(B' R) B`
        B_q_transformed = torch.matmul(B_q_rot, B_q_2)  # [seq_len, num_heads*32, num_heads*32]
        q_h = q_concat.to(torch.float16)  # [batch, seq_len, num_heads*32]
        # `q' = (B' R B) q`
        q_final = torch.einsum("bsi,sij->bsj", q_h, B_q_transformed)  # [batch, seq_len, num_heads*32]
        q_transformed += q_final

    for i in range(head_dim_half * num_heads_k):
        rope_matrix_h = cos_size_matrix[:, i%head_dim_half, :, :]  # [seq_len, 2, 2]
        B_k_2 = B_k[2*i:2*(i+1), :].to(torch.float16)  # [2, num_heads*32]
        # `B' * R`
        B_k_rot = torch.matmul(B_k_2.T.to(torch.float16), rope_matrix_h.to(torch.float16))  # 
        # `(B' R) B`
        B_k_transformed = torch.matmul(B_k_rot, B_k_2)  # [seq_len, num_heads*32, num_heads*32]
        k_h = k_concat.to(torch.float16)  # [batch, seq_len, num_heads*32]
        # `q' = (B' R B) q`
        k_final = torch.einsum("bsi,sij->bsj", k_h, B_k_transformed)  # [batch, seq_len, num_heads*32]
        k_transformed += k_final

    q_embed = q_transformed.view(batch_size, seq_len, num_heads_q * head_dim)
    k_embed = k_transformed.view(batch_size, seq_len, num_heads_k * head_dim)

    return q_embed, k_embed

def apply_rotary_pos_emb_hla_fast(q, k, cos_size_matrix, B_q, B_k):
    # print("q shape: ", q.shape)
    # print("k shape: ", k.shape)
    # print("cos_size_matrix shape: ", cos_size_matrix.shape)
    # print("B_q shape: ", B_q.shape)
    # print("B_k shape: ", B_k.shape)
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
        B_blocks = B_blocks.unsqueeze(0).expand(seq_len, -1, -1, -1)  # [s, n_blocks, 2, D]
        
        # Expand cos matrix to [seq_len, num_blocks, 2, 2]
        cos_expanded = cos_matrix[:, :head_dim_half]  # [s, 16, 2, 2]
        cos_expanded = cos_expanded.unsqueeze(1)  # [s, 1, 16, 2, 2]
        cos_expanded = cos_expanded.expand(-1, num_heads, -1, -1, -1)  # [s, nh, 16, 2, 2]
        cos_expanded = cos_expanded.reshape(seq_len, num_blocks, 2, 2)  # [s, n_blocks, 2, 2]

        # Calculate B'R [s, n_blocks, D, 2]
        B_rot = torch.einsum('snij,snjk->snik', 
                            B_blocks.transpose(2,3).to(torch.float16),  # [s,n_blocks,D,2]
                            cos_expanded.to(torch.float16))              # [s,n_blocks,2,2]
        
        # Calculate (B'R)B [s, n_blocks, D, D]
        B_trans = torch.einsum('snik,snkj->snij',  # Key dimension alignment fix
                             B_rot,                # [s,n_blocks,D,2]
                             B_blocks.to(torch.float16))  # [s,n_blocks,2,D]
        
        # Apply transformation and accumulate [batch, seq_len, D]
        x_trans = torch.einsum('bsd,snij->bsnj', 
                              x.to(torch.float16),  # [batch, s, D]
                              B_trans)              # [s,n_blocks,D,D]
        return x_trans.sum(dim=2).to(x.dtype)       # Sum along block dimension

    # Execute transformations
    q_transformed = parallel_transform(q_concat, B_q, num_heads_q, cos_size_matrix)
    k_transformed = parallel_transform(k_concat, B_k, num_heads_k, cos_size_matrix)

    # Restore original shape [batch, num_heads, seq_len, head_dim]
    q_embed = q_transformed.view(batch_size, seq_len, num_heads_q* head_dim)
    k_embed = k_transformed.view(batch_size, seq_len, num_heads_k* head_dim)

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

def benchmark(func, q, k, cos, B_q, B_k, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = func(q, k, cos, B_q, B_k)
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        _ = func(q, k, cos, B_q, B_k)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs



if __name__ == '__main__':
    q = torch.randn(1, 6, 5, 32, device="cuda")   # 原函数输入
    k = torch.randn(1, 2, 5, 32, device="cuda")
    cos_size_matrix = torch.randn(5, 16, 2, 2, device="cuda")
    B_q = torch.randn(192, 192, device="cuda")
    B_k = torch.randn(64, 64, device="cuda")
    # 测试原始版本
    t1 = benchmark(apply_rotary_pos_emb_hla, q, k, cos_size_matrix, B_q, B_k)

    # 测试优化版本
    t2 = benchmark(apply_rotary_pos_emb_hla_fast, q, k, cos_size_matrix, B_q, B_k)

    print(f"原始版本: {t1:.3f} ms/次")
    print(f"优化版本: {t2:.3f} ms/次")
    print(f"速度提升: {t1/t2:.1f}x")