import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, LlamaConfig
from modeling_llama_HLA import LlamaRotaryEmbeddingmy, LlamaRotaryEmbedding, apply_rotary_pos_emb
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
                            cos_expanded)              # [s,n_blocks,2,2]
        
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
class RoPEBenchmark:
    def __init__(self, model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = LlamaConfig.from_pretrained(model_name)

        # Initialize both RoPE implementations
        self.rope_original = LlamaRotaryEmbedding(self.config).to(self.device)
        self.rope_new = LlamaRotaryEmbeddingmy(self.config).to(self.device)

        # Create B matrices for head-level adaptation
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_heads = self.config.num_attention_heads
        num_heads_kv = self.config.num_key_value_heads
        total_dim_q = num_heads * head_dim
        total_dim_kv = num_heads_kv * head_dim

        # Initialize random B matrices for testing
        self.B_q = torch.randn(total_dim_q // 2, total_dim_q // 2).to(self.device)
        self.B_k = torch.randn(total_dim_kv // 2, total_dim_kv //2).to(self.device)

    def generate_inputs(self, batch_size, seq_len):
        """Generate random inputs for benchmarking"""
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).to(self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create query and key tensors
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_heads_q = self.config.num_attention_heads
        num_heads_kv = self.config.num_key_value_heads

        q = torch.randn(batch_size, num_heads_q, seq_len, head_dim).to(self.device)
        k = torch.randn(batch_size, num_heads_kv, seq_len, head_dim).to(self.device)
        q_l = torch.randn(batch_size, num_heads_q, seq_len, head_dim // 2).to(self.device)
        k_l = torch.randn(batch_size, num_heads_kv, seq_len, head_dim // 2).to(self.device)

        return q, k, q_l, k_l, position_ids

    def benchmark_single_run(self, batch_size, seq_len, n_iterations=10, warmup=3):
        """Run benchmark for a specific batch size and sequence length"""
        q, k, q_l, k_l, position_ids = self.generate_inputs(batch_size, seq_len)

        # Warm-up runs
        for _ in range(warmup):
            # Original implementation
            cos_orig, sin_orig = self.rope_original(q, position_ids)
            q_embed_orig, k_embed_orig = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)

            # New implementation
            cos_size_matrix = self.rope_new(q, position_ids)
            q_embed_new, k_embed_new = apply_rotary_pos_emb_hla_fast(q_l, k_l, cos_size_matrix, self.B_q, self.B_k)

        # Benchmark original implementation
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        for _ in range(n_iterations):
            cos_orig, sin_orig = self.rope_original(q, position_ids)
            q_embed_orig, k_embed_orig = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)
            torch.cuda.synchronize() if self.device == "cuda" else None
        orig_time = (time.time() - start_time) / n_iterations

        # Benchmark new implementation
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        for _ in range(n_iterations):
            cos_size_matrix = self.rope_new(q, position_ids)
            q_embed_new, k_embed_new = apply_rotary_pos_emb_hla_fast(q_l, k_l, cos_size_matrix, self.B_q, self.B_k)
            torch.cuda.synchronize() if self.device == "cuda" else None
        new_time = (time.time() - start_time) / n_iterations

        # Calculate memory usage
        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        cos_orig, sin_orig = self.rope_original(q, position_ids)
        q_embed_orig, k_embed_orig = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)
        orig_memory = torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else 0

        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        cos_size_matrix = self.rope_new(q, position_ids)
        q_embed_new, k_embed_new = apply_rotary_pos_emb_hla_fast(q_l, k_l, cos_size_matrix, self.B_q, self.B_k)
        new_memory = torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else 0

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "orig_time": orig_time * 1000,  # Convert to ms
            "new_time": new_time * 1000,    # Convert to ms
            "orig_memory": orig_memory,
            "new_memory": new_memory,
            "speedup": orig_time / new_time if new_time > 0 else float('inf'),
            "memory_ratio": new_memory / orig_memory if orig_memory > 0 else float('inf')
        }

    def run_benchmark(self, batch_sizes=[1, 4, 8], seq_lengths=[128, 512, 1024, 2048], n_iterations=10):
        """Run benchmark across different batch sizes and sequence lengths"""
        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
                result = self.benchmark_single_run(batch_size, seq_len, n_iterations)
                results.append(result)
                print(f"  Original: {result['orig_time']:.2f}ms, {result['orig_memory']:.2f}MB")
                print(f"  New: {result['new_time']:.2f}ms, {result['new_memory']:.2f}MB")
                print(f"  Speedup: {result['speedup']:.2f}x, Memory ratio: {result['memory_ratio']:.2f}x")

        return results

    def verify_outputs(self, batch_size=1, seq_len=128):
        """Verify that both implementations produce similar outputs"""
        q, k, q_l, k_l, position_ids = self.generate_inputs(batch_size, seq_len)

        # Original implementation
        cos_orig, sin_orig = self.rope_original(q, position_ids)
        q_embed_orig, k_embed_orig = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)

        # New implementation
        cos_size_matrix = self.rope_new(q, position_ids)
        q_embed_new, k_embed_new = apply_rotary_pos_emb_hla_fast(q_l, k_l, cos_size_matrix, self.B_q, self.B_k)

        # Reshape new outputs to match original shape
        q_embed_new_reshaped = q_embed_new.view(batch_size, seq_len, self.config.num_attention_heads, -1).permute(0, 2, 1, 3)
        k_embed_new_reshaped = k_embed_new.view(batch_size, seq_len, self.config.num_attention_heads, -1).permute(0, 2, 1, 3)

        # Calculate differences
        q_diff = torch.abs(q_embed_orig - q_embed_new_reshaped).mean().item()
        k_diff = torch.abs(k_embed_orig - k_embed_new_reshaped).mean().item()

        return {
            "q_diff": q_diff,
            "k_diff": k_diff,
            "output_similar": q_diff < 1e-3 and k_diff < 1e-3
        }

    def plot_results(self, results):
        """Plot benchmark results"""
        batch_sizes = sorted(list(set([r["batch_size"] for r in results])))
        seq_lengths = sorted(list(set([r["seq_len"] for r in results])))

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Time comparison
        ax = axes[0]
        width = 0.35
        x = np.arange(len(seq_lengths))

        for i, batch_size in enumerate(batch_sizes):
            batch_results = [r for r in results if r["batch_size"] == batch_size]
            batch_results.sort(key=lambda r: r["seq_len"])

            orig_times = [r["orig_time"] for r in batch_results]
            new_times = [r["new_time"] for r in batch_results]

            offset = width * (i - len(batch_sizes)/2 + 0.5)
            ax.bar(x + offset - width/4, orig_times, width/2, label=f"Original (bs={batch_size})")
            ax.bar(x + offset + width/4, new_times, width/2, label=f"New (bs={batch_size})")

        ax.set_ylabel('Time (ms)')
        ax.set_xlabel('Sequence Length')
        ax.set_title('Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lengths)
        ax.legend()

        # Memory comparison
        ax = axes[1]
        for i, batch_size in enumerate(batch_sizes):
            batch_results = [r for r in results if r["batch_size"] == batch_size]
            batch_results.sort(key=lambda r: r["seq_len"])

            orig_memory = [r["orig_memory"] for r in batch_results]
            new_memory = [r["new_memory"] for r in batch_results]

            offset = width * (i - len(batch_sizes)/2 + 0.5)
            ax.bar(x + offset - width/4, orig_memory, width/2, label=f"Original (bs={batch_size})")
            ax.bar(x + offset + width/4, new_memory, width/2, label=f"New (bs={batch_size})")

        ax.set_ylabel('Memory (MB)')
        ax.set_xlabel('Sequence Length')
        ax.set_title('Memory Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lengths)
        ax.legend()

        plt.tight_layout()
        plt.savefig('rope_benchmark_results.png')
        plt.show()

def main():
    # Initialize benchmark
    benchmark = RoPEBenchmark()

    # Verify output similarity
    # print("Verifying output similarity...")
    # similarity = benchmark.verify_outputs()
    # print(f"Output differences - Q: {similarity['q_diff']:.6f}, K: {similarity['k_diff']:.6f}")
    # print(f"Outputs similar: {similarity['output_similar']}")

    # Run benchmarks
    print("\nRunning benchmarks...")
    results = benchmark.run_benchmark(
        batch_sizes=[1, 4, 8],
        seq_lengths=[128, 512, 1024, 2048],
        n_iterations=5
    )

    # Plot results
    benchmark.plot_results(results)

    # Print summary
    print("\nSummary:")
    for batch_size in [1]:
        for seq_len in [128, 512, 1024, 2048]:
            result = next((r for r in results if r["batch_size"] == batch_size and r["seq_len"] == seq_len), None)
            if result:
                print(f"Batch={batch_size}, Seq={seq_len}: Speedup={result['speedup']:.2f}x, Memory={result['memory_ratio']:.2f}x")

if __name__ == "__main__":
    main()