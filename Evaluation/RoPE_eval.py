import torch
import time
import gc
import numpy as np
import psutil
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from modeling_llama_HLA import LlamaRotaryEmbeddingmy, LlamaRotaryEmbedding
from transformers import AutoTokenizer, LlamaConfig

class RoPEBenchmark:
    def __init__(self, model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = LlamaConfig.from_pretrained(model_name)
        self.process = psutil.Process(os.getpid())
    
    def _clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def _get_current_memory(self):
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        return self.process.memory_info().rss / (1024 ** 2)  # MB
    
    def _time_function(self, func, *args, trials=50, inner_loops=100):
        """
        Times inner_loops calls to func for each trial.
        Returns the median per-call time and its standard deviation.
        """
        times = []
        # Warmup iterations
        for _ in range(5):
            with torch.no_grad():
                for _ in range(inner_loops):
                    _ = func(*args)
        # Timing trials (using CUDA events for precision if on GPU)
        if self.device == "cuda":
            for _ in range(trials):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    for _ in range(inner_loops):
                        _ = func(*args)
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event) / 1000.0  # convert ms to seconds
                per_call_time = total_time / inner_loops
                times.append(per_call_time)
        else:
            for _ in range(trials):
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(inner_loops):
                        _ = func(*args)
                total_time = time.time() - start_time
                per_call_time = total_time / inner_loops
                times.append(per_call_time)
        median_time = np.median(times)
        error = np.std(times)
        return median_time, error

    def benchmark_rope(self, batch_sizes=[1], seq_lengths=[512, 1024, 2048, 4096],
                       head_dim=32, num_heads=32, trials=50, inner_loops=100):
        """
        Runs a single benchmark session for standalone RoPE.
        Returns a dictionary with median timing and memory usage for both implementations.
        """
        results = {"original": {"time": [], "memory": []},
                   "hla": {"time": [], "memory": []}}
        configs = [(batch, seq) for batch in batch_sizes for seq in seq_lengths]
        
        for batch_size, seq_len in configs:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
            # Prepare test inputs
            hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim,
                                        dtype=torch.float16, device=self.device)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Benchmark Original RoPE
            self._clear_gpu_memory()
            start_mem = self._get_current_memory()
            original_rope = LlamaRotaryEmbedding(config=self.config).to(self.device)
            orig_time, orig_error = self._time_function(original_rope, hidden_states, position_ids,
                                                        trials=trials, inner_loops=inner_loops)
            end_mem = self._get_current_memory()
            orig_mem_usage = end_mem - start_mem
            results["original"]["time"].append((batch_size, seq_len, orig_time, orig_error))
            results["original"]["memory"].append((batch_size, seq_len, orig_mem_usage))
            # print(f"Original RoPE: median {orig_time*1000:.2f} ms ± {orig_error*1000:.2f} ms, Memory diff: {orig_mem_usage:.2f} MB")
            
            # Benchmark HLA RoPE
            self._clear_gpu_memory()
            start_mem = self._get_current_memory()
            hla_rope = LlamaRotaryEmbeddingmy(config=self.config).to(self.device)
            hla_time, hla_error = self._time_function(hla_rope, hidden_states, position_ids,
                                                      trials=trials, inner_loops=inner_loops)
            end_mem = self._get_current_memory()
            hla_mem_usage = end_mem - start_mem
            results["hla"]["time"].append((batch_size, seq_len, hla_time, hla_error))
            results["hla"]["memory"].append((batch_size, seq_len, hla_mem_usage))
            # print(f"HLA RoPE: median {hla_time*1000:.2f} ms ± {hla_error*1000:.2f} ms, Memory diff: {hla_mem_usage:.2f} MB")
            
            speedup = orig_time / hla_time if hla_time > 0 else float('inf')
            # print(f"Speedup (HLA vs Original): {speedup:.2f}x")
        
        return results

    def run_multiple_sessions(self, sessions=30, **kwargs):
        """
        Run the benchmark multiple times and aggregate the results.
        """
        all_results = []
        for i in range(sessions):
            print(f"\n=== Benchmark Session {i+1}/{sessions} ===")
            session_result = self.benchmark_rope(**kwargs)
            all_results.append(session_result)
        
        # Aggregate results per configuration (keyed by (batch_size, seq_length))
        aggregated = {"original": {"time": {}, "memory": {}},
                      "hla": {"time": {}, "memory": {}}}
        for method in ["original", "hla"]:
            for res_type in ["time", "memory"]:
                for session in all_results:
                    for tup in session[method][res_type]:
                        key = (tup[0], tup[1])
                        aggregated[method][res_type].setdefault(key, []).append(tup[2])
        # Compute aggregated median and standard deviation
        aggregated_results = {"original": {"time": [], "memory": []},
                              "hla": {"time": [], "memory": []}}
        for method in ["original", "hla"]:
            for key, values in aggregated[method]["time"].items():
                agg_median = np.median(values)
                agg_std = np.std(values)
                aggregated_results[method]["time"].append((key[0], key[1], agg_median, agg_std))
            for key, values in aggregated[method]["memory"].items():
                agg_mem = np.median(values)
                aggregated_results[method]["memory"].append((key[0], key[1], agg_mem))
        return aggregated_results

    def profile_rope_comparison_multi(self, batch_size=1, seq_len=2048, head_dim=32, num_heads=32, iterations=10):
        """
        Profiles both Original and HLA RoPE implementations over multiple iterations.
        Aggregates (via median) the total CPU and CUDA time across runs.
        Returns a dictionary with aggregated profiling metrics.
        """
        cpu_times_orig = []
        cuda_times_orig = []
        cpu_times_hla = []
        cuda_times_hla = []
        
        # Prepare test inputs
        hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim,
                                    dtype=torch.float16, device=self.device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        for _ in range(iterations):
            # Profile Original RoPE
            original_rope = LlamaRotaryEmbedding(config=self.config).to(self.device)
            _ = original_rope(hidden_states, position_ids)  # Warmup
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == "cuda" else [ProfilerActivity.CPU],
                        record_shapes=False, profile_memory=False) as prof_orig:
                with record_function("original_rope"):
                    _ = original_rope(hidden_states, position_ids)
            key_avg_orig = prof_orig.key_averages()
            total_cpu_orig = sum([item.cpu_time_total for item in key_avg_orig])
            total_cuda_orig = sum([getattr(item, "cuda_time_total", 0) for item in key_avg_orig]) if self.device == "cuda" else 0
            cpu_times_orig.append(total_cpu_orig)
            cuda_times_orig.append(total_cuda_orig)
            
            # Profile HLA RoPE
            hla_rope = LlamaRotaryEmbeddingmy(config=self.config).to(self.device)
            _ = hla_rope(hidden_states, position_ids)  # Warmup
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == "cuda" else [ProfilerActivity.CPU],
                        record_shapes=False, profile_memory=False) as prof_hla:
                with record_function("hla_rope"):
                    _ = hla_rope(hidden_states, position_ids)
            key_avg_hla = prof_hla.key_averages()
            total_cpu_hla = sum([item.cpu_time_total for item in key_avg_hla])
            total_cuda_hla = sum([getattr(item, "cuda_time_total", 0) for item in key_avg_hla]) if self.device == "cuda" else 0
            cpu_times_hla.append(total_cpu_hla)
            cuda_times_hla.append(total_cuda_hla)
        
        aggregated_metrics = {
            "original": {
                "total_cpu_time": np.median(cpu_times_orig),
                "total_cuda_time": np.median(cuda_times_orig) if self.device == "cuda" else None
            },
            "hla": {
                "total_cpu_time": np.median(cpu_times_hla),
                "total_cuda_time": np.median(cuda_times_hla) if self.device == "cuda" else None
            }
        }
        return aggregated_metrics


    def plot_profiling_comparison(self, metrics):
        """
        Plots a bar chart comparing aggregated total CPU and CUDA times (converted from µs to ms)
        between Original and HLA RoPE implementations.
        """
        methods = ["original", "hla"]
        cpu_times = [metrics[m]["total_cpu_time"] for m in methods]
        cpu_times_ms = [t / 1000.0 for t in cpu_times]
        if self.device == "cuda":
            cuda_times = [metrics[m]["total_cuda_time"] for m in methods]
            cuda_times_ms = [t / 1000.0 for t in cuda_times]
        else:
            cuda_times_ms = [0, 0]
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].bar(methods, cpu_times_ms, color=['blue', 'orange'])
        ax[0].set_title("Aggregated Total CPU Time (ms)")
        ax[0].set_ylabel("Time (ms)")
        if self.device == "cuda":
            ax[1].bar(methods, cuda_times_ms, color=['blue', 'orange'])
            ax[1].set_title("Aggregated Total CUDA Time (ms)")
            ax[1].set_ylabel("Time (ms)")
        plt.tight_layout()
        plt.savefig("profiling_comparison.png")
        plt.show()

    def plot_results(self, results):
        """
        Plots aggregated benchmark results (execution time and memory usage) versus sequence length.
        """
        # Group results by sequence length (assumes same batch size for simplicity)
        seq_lengths = sorted(set(x[1] for x in results["original"]["time"]))
        orig_time = []
        orig_std = []
        hla_time = []
        hla_std = []
        orig_mem = []
        hla_mem = []
        
        for seq in seq_lengths:
            orig_entries = [t for t in results["original"]["time"] if t[1] == seq]
            hla_entries = [t for t in results["hla"]["time"] if t[1] == seq]
            orig_mem_entries = [m for m in results["original"]["memory"] if m[1] == seq]
            hla_mem_entries = [m for m in results["hla"]["memory"] if m[1] == seq]
            
            if orig_entries:
                times = [entry[2] for entry in orig_entries]
                stds = [entry[3] for entry in orig_entries]
                orig_time.append(np.median(times)*1000)  # convert s to ms
                orig_std.append(np.median(stds)*1000)
            else:
                orig_time.append(0)
                orig_std.append(0)
            if hla_entries:
                times = [entry[2] for entry in hla_entries]
                stds = [entry[3] for entry in hla_entries]
                hla_time.append(np.median(times)*1000)
                hla_std.append(np.median(stds)*1000)
            else:
                hla_time.append(0)
                hla_std.append(0)
            if orig_mem_entries:
                mems = [entry[2] for entry in orig_mem_entries]
                orig_mem.append(np.median(mems))
            else:
                orig_mem.append(0)
            if hla_mem_entries:
                mems = [entry[2] for entry in hla_mem_entries]
                hla_mem.append(np.median(mems))
            else:
                hla_mem.append(0)
        
        # Plot execution time with error bars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.errorbar(seq_lengths, orig_time, yerr=orig_std, label="Original RoPE", marker='o', capsize=5)
        ax1.errorbar(seq_lengths, hla_time, yerr=hla_std, label="HLA RoPE", marker='o', capsize=5)
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_title("Aggregated RoPE Execution Time (Median)")
        ax1.legend()
        ax1.grid(True)
        
        # Plot memory usage
        ax2.plot(seq_lengths, orig_mem, label="Original RoPE", marker='o')
        ax2.plot(seq_lengths, hla_mem, label="HLA RoPE", marker='o')
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_title("Aggregated RoPE Memory Usage (Median)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("aggregated_rope_benchmark_results.png")
        plt.show()

if __name__ == "__main__":
    # Create benchmark instance
    benchmark = RoPEBenchmark(model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer")
    
    # Run multiple benchmark sessions for aggregated results
    sessions = 20  # For robust statistical aggregation
    print(f"\n=== Running {sessions} Benchmark Sessions ===")
    aggregated_results = benchmark.run_multiple_sessions(
        sessions=sessions,
        batch_sizes=[1],
        seq_lengths=[512, 1024, 2048],
        trials=30,
        inner_loops=100
    )
    
    print("\nAggregated Benchmark Results:")
    for method in ["original", "hla"]:
        for tup in aggregated_results[method]["time"]:
            print(f"{method} - Batch: {tup[0]}, Seq: {tup[1]}, Median Time: {tup[2]*1000:.2f} ms ± {tup[3]*1000:.2f} ms")
        for tup in aggregated_results[method]["memory"]:
            print(f"{method} - Batch: {tup[0]}, Seq: {tup[1]}, Memory Usage: {tup[2]:.2f} MB")
    
    # Plot aggregated benchmark results
    benchmark.plot_results(aggregated_results)
    
    # Run multiple profiling iterations for a chosen configuration and aggregate the results
    print("\n=== Running Detailed Profiling (Multiple Iterations) for batch_size=1, seq_len=2048 ===")
    profiling_metrics = benchmark.profile_rope_comparison_multi(batch_size=1, seq_len=2048, iterations=10)
    benchmark.plot_profiling_comparison(profiling_metrics)
    
    print("\nBenchmarking and profiling completed. Check the generated PNG files for graphs.")
