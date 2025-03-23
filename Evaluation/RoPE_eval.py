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
        
        # Enable CUDA benchmark mode for optimized performance on fixed-size inputs
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
    
    def _clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
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
        # More thorough warmup to ensure GPU is properly initialized
        for _ in range(10):  # Increased warmup iterations
            with torch.no_grad():
                for _ in range(inner_loops):
                    result = func(*args)
                    if self.device == "cuda":
                        torch.cuda.synchronize()  # Ensure operation completion
        
        if self.device == "cuda":
            for _ in range(trials):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Clear cache between trials for consistent measurement
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                start_event.record()
                with torch.no_grad():
                    for _ in range(inner_loops):
                        result = func(*args)
                        # Force result to be computed (prevent lazy evaluation)
                        _ = sum(t.sum() for t in result).item()
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
                times.append(total_time / inner_loops)
        else:
            for _ in range(trials):
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(inner_loops):
                        _ = func(*args)
                total_time = time.time() - start_time
                times.append(total_time / inner_loops)
        
        return np.median(times), np.std(times)

    def benchmark_rope(self, batch_sizes=[1, 8], seq_lengths=[512, 1024, 2048, 4096],
                       head_dim=64, num_heads=6, trials=50, inner_loops=100):
        """
        Runs a benchmark for standalone RoPE implementations.
        Returns a dictionary with median timing and memory usage for both methods.
        Increased batch sizes and dimensions for better GPU utilization.
        """
        results = {"original": {"time": [], "memory": []},
                   "hla": {"time": [], "memory": []}}
        configs = [(batch, seq) for batch in batch_sizes for seq in seq_lengths]
        
        for batch_size, seq_len in configs:
            print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
            # Prepare test inputs on device with larger dimensions for better GPU utilization
            hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim,
                                        dtype=torch.float16, device=self.device)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Ensure tensors are actually on GPU
            if self.device == "cuda":
                assert hidden_states.is_cuda, "Hidden states not on CUDA"
                assert position_ids.is_cuda, "Position ids not on CUDA"
            
            # Benchmark Original RoPE
            self._clear_gpu_memory()
            start_mem = self._get_current_memory()
            original_rope = LlamaRotaryEmbedding(config=self.config).to(self.device)
            
            # Run warmup iteration and verify GPU usage
            with torch.no_grad():
                warmup_result = original_rope(hidden_states, position_ids)
                if self.device == "cuda":
                    assert warmup_result[0].is_cuda, "RoPE output not on CUDA"
                    torch.cuda.synchronize()
                    print(f"  Original RoPE GPU memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            orig_time, orig_error = self._time_function(original_rope, hidden_states, position_ids,
                                                        trials=trials, inner_loops=inner_loops)
            end_mem = self._get_current_memory()
            orig_mem_usage = end_mem - start_mem
            results["original"]["time"].append((batch_size, seq_len, orig_time, orig_error))
            results["original"]["memory"].append((batch_size, seq_len, orig_mem_usage))
            
            # Benchmark HLA RoPE
            self._clear_gpu_memory()
            start_mem = self._get_current_memory()
            hla_rope = LlamaRotaryEmbeddingmy(config=self.config).to(self.device)
            
            # Run warmup iteration and verify GPU usage
            with torch.no_grad():
                warmup_result = hla_rope(hidden_states, position_ids)
                if self.device == "cuda":
                    assert warmup_result.is_cuda, "HLA RoPE output not on CUDA"
                    torch.cuda.synchronize()
                    print(f"  HLA RoPE GPU memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            hla_time, hla_error = self._time_function(hla_rope, hidden_states, position_ids,
                                                      trials=trials, inner_loops=inner_loops)
            end_mem = self._get_current_memory()
            hla_mem_usage = end_mem - start_mem
            results["hla"]["time"].append((batch_size, seq_len, hla_time, hla_error))
            results["hla"]["memory"].append((batch_size, seq_len, hla_mem_usage))
            
            speedup = orig_time / hla_time if hla_time > 0 else float('inf')
            print(f"Speedup (HLA vs Original): {speedup:.2f}x")
            
            # Report GPU utilization if available
            if self.device == "cuda":
                print(f"  GPU utilization: {torch.cuda.utilization()}%")
        
        return results

    def run_multiple_sessions(self, sessions=5, **kwargs):
        """
        Runs the benchmark multiple times and aggregates results.
        Reduced number of sessions to avoid CUDA OOM errors.
        """
        all_results = []
        for i in range(sessions):
            print(f"\n=== Benchmark Session {i+1}/{sessions} ===")
            session_result = self.benchmark_rope(**kwargs)
            all_results.append(session_result)
            
            # Add explicit cooling period between sessions
            if self.device == "cuda" and i < sessions - 1:
                print("Cooling GPU between sessions...")
                self._clear_gpu_memory()
                time.sleep(2)  # Allow GPU to cool down
        
        # Aggregate results per configuration (keyed by (batch_size, seq_length))
        aggregated = {"original": {"time": {}, "memory": {}},
                      "hla": {"time": {}, "memory": {}}}
        for method in ["original", "hla"]:
            for res_type in ["time", "memory"]:
                for session in all_results:
                    for tup in session[method][res_type]:
                        key = (tup[0], tup[1])
                        aggregated[method][res_type].setdefault(key, []).append(tup[2])
        
        aggregated_results = {"original": {"time": [], "memory": []},
                              "hla": {"time": [], "memory": []}}
        for method in ["original", "hla"]:
            for key, values in aggregated[method]["time"].items():
                aggregated_results[method]["time"].append((key[0], key[1], np.median(values), np.std(values)))
            for key, values in aggregated[method]["memory"].items():
                aggregated_results[method]["memory"].append((key[0], key[1], np.median(values)))
        return aggregated_results

    def profile_rope_comparison_multi(self, batch_size=8, seq_len=2048, head_dim=64, num_heads=32, iterations=10):
        """
        Profiles both Original and HLA RoPE implementations over multiple iterations.
        Uses torch.profiler for detailed profiling and explicit CUDA events for HLA RoPE timing.
        """
        cpu_times_orig = []
        cuda_times_orig = []
        cpu_times_hla = []
        cuda_times_hla = []
        
        hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim,
                                    dtype=torch.float16, device=self.device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        if self.device == "cuda":
            print(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        
        manual_hla_cuda_time_ms = 0

        for i in range(iterations):
            print(f"\nProfiling iteration {i+1}/{iterations}")
            
            # --- Profile Original RoPE ---
            original_rope = LlamaRotaryEmbedding(config=self.config).to(self.device)
            
            # Warmup for original_rope
            for _ in range(5):
                with torch.no_grad():
                    warmup_result = original_rope(hidden_states, position_ids)
                    if self.device == "cuda":
                        _ = sum(t.sum() for t in warmup_result).item()
                        torch.cuda.synchronize()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Profile original_rope with increased loop iterations to accumulate CUDA time
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == "cuda" else [ProfilerActivity.CPU],
                        record_shapes=True, profile_memory=True, with_stack=True, with_flops=True) as prof_orig:
                with record_function("original_rope"):
                    with torch.no_grad():
                        for _ in range(100):
                            result = original_rope(hidden_states, position_ids)
                            if self.device == "cuda":
                                _ = sum(t.sum() for t in result).item()
                        if self.device == "cuda":
                            torch.cuda.synchronize()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            events_orig = prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            print(f"Original RoPE Profile Events:\n{events_orig}")
            
            key_avg_orig = prof_orig.key_averages()
            total_cpu_orig = sum(item.cpu_time_total for item in key_avg_orig)
            total_cuda_orig = sum(getattr(item, "cuda_time_total", 0) for item in key_avg_orig) if self.device == "cuda" else 0
            cpu_times_orig.append(total_cpu_orig)
            cuda_times_orig.append(total_cuda_orig)
            
            if self.device == "cuda":
                print(f"  Original RoPE peak memory: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
                print(f"  Original RoPE CUDA time (from profiler): {total_cuda_orig:.4f} μs")
            
            self._clear_gpu_memory()
            
            # --- Profile HLA RoPE ---
            hla_rope = LlamaRotaryEmbeddingmy(config=self.config).to(self.device)
            
            # Warmup for hla_rope
            for _ in range(5):
                with torch.no_grad():
                    warmup_result = hla_rope(hidden_states, position_ids)
                    if self.device == "cuda":
                        _ = warmup_result.sum().item()  # hla_rope returns a tensor
                        torch.cuda.synchronize()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Use explicit CUDA events for manual timing of hla_rope
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == "cuda" else [ProfilerActivity.CPU],
                        record_shapes=True, profile_memory=True, with_stack=True, with_flops=True) as prof_hla:
                with record_function("hla_rope"):
                    with torch.no_grad():
                        for _ in range(100):
                            result = hla_rope(hidden_states, position_ids)
                            _ = result.sum().item()
                        if self.device == "cuda":
                            torch.cuda.synchronize()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                end_event.record()
                torch.cuda.synchronize()
                manual_hla_cuda_time_ms = start_event.elapsed_time(end_event)
                print(f"HLA RoPE measured CUDA elapsed time (manual timing): {manual_hla_cuda_time_ms:.4f} ms")
            else:
                manual_hla_cuda_time_ms = 0
            
            events_hla = prof_hla.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            print(f"HLA RoPE Profile Events:\n{events_hla}")
            
            key_avg_hla = prof_hla.key_averages()
            total_cpu_hla = sum(item.cpu_time_total for item in key_avg_hla)
            total_cuda_hla = sum(getattr(item, "cuda_time_total", 0) for item in key_avg_hla) if self.device == "cuda" else 0
            cpu_times_hla.append(total_cpu_hla)
            cuda_times_hla.append(total_cuda_hla)
            
            if self.device == "cuda":
                print(f"  HLA RoPE peak memory: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
                print(f"  HLA RoPE CUDA time (from profiler): {total_cuda_hla:.4f} μs")
            
            self._clear_gpu_memory()
        
        # Use filtered values for median calculation if available
        cuda_times_orig_filtered = [t for t in cuda_times_orig if t > 0]
        cuda_times_hla_filtered = [t for t in cuda_times_hla if t > 0]
        
        cuda_orig_median = np.median(cuda_times_orig_filtered) if cuda_times_orig_filtered else np.median(cuda_times_orig)
        cuda_hla_median = np.median(cuda_times_hla_filtered) if cuda_times_hla_filtered else np.median(cuda_times_hla)
        
        aggregated_metrics = {
            "original": {
                "total_cpu_time": np.median(cpu_times_orig),
                "total_cuda_time": cuda_orig_median if self.device == "cuda" else None
            },
            "hla": {
                "total_cpu_time": np.median(cpu_times_hla),
                "total_cuda_time": cuda_hla_median if self.device == "cuda" else None,
                "manual_cuda_time_ms": manual_hla_cuda_time_ms if self.device == "cuda" else None
            }
        }
        
        if self.device == "cuda":
            print("\nCUDA Operations Profile Summary:")
            print("Original RoPE CUDA Time (from profiler): {:.4f} μs".format(aggregated_metrics["original"]["total_cuda_time"]))
            print("HLA RoPE CUDA Time (from profiler): {:.4f} μs".format(aggregated_metrics["hla"]["total_cuda_time"]))
            print("HLA RoPE CUDA Time (manual timing): {:.4f} ms".format(aggregated_metrics["hla"]["manual_cuda_time_ms"]))
            
            if aggregated_metrics["original"]["total_cuda_time"] > 0 and aggregated_metrics["hla"]["total_cuda_time"] > 0:
                speedup = aggregated_metrics["original"]["total_cuda_time"] / aggregated_metrics["hla"]["total_cuda_time"]
                print("Speedup (CUDA): {:.2f}x".format(speedup))
            else:
                print("Speedup (CUDA): Unable to calculate (one or both times are zero)")
        
        return aggregated_metrics

    
    def plot_profiling_comparison(self, metrics):
        """
        Plots a bar chart comparing aggregated CPU and CUDA times in microseconds.
        """
        methods = ["original", "hla"]
        cpu_times = [metrics[m]["total_cpu_time"] for m in methods]
        # Convert to microseconds (original times are in nanoseconds)
        cpu_times_us = [t / 1000.0 for t in cpu_times]
        
        fig, axs = plt.subplots(1, 2 if self.device == "cuda" else 1, figsize=(12, 6))
        
        if self.device == "cuda":
            cuda_times = [metrics[m]["total_cuda_time"] for m in methods]
            # Convert to microseconds
            cuda_times_us = [t / 1000.0 for t in cuda_times]
            
            axs[0].bar(methods, cpu_times_us, color=['blue', 'orange'])
            axs[0].set_title("Total CPU Time (μs)")
            axs[0].set_ylabel("Time (μs)")
            
            axs[1].bar(methods, cuda_times_us, color=['blue', 'orange'])
            axs[1].set_title("Total CUDA Time (μs)")
            axs[1].set_ylabel("Time (μs)")
        else:
            axs.bar(methods, cpu_times_us, color=['blue', 'orange'])
            axs.set_title("Total CPU Time (μs)")
            axs.set_ylabel("Time (μs)")
        
        plt.tight_layout()
        plt.savefig("profiling_comparison_microseconds.png")
        plt.show()

    def plot_results(self, results):
        """
        Plots aggregated benchmark results (execution time and memory usage) versus sequence length.
        """
        # Group by batch size
        batch_sizes = sorted(set(x[0] for x in results["original"]["time"]))
        
        for batch_size in batch_sizes:
            seq_lengths = sorted(set(x[1] for x in results["original"]["time"] if x[0] == batch_size))
            orig_time, orig_std, hla_time, hla_std = [], [], [], []
            orig_mem, hla_mem = [], []
            
            for seq in seq_lengths:
                orig_entries = [t for t in results["original"]["time"] if t[0] == batch_size and t[1] == seq]
                hla_entries = [t for t in results["hla"]["time"] if t[0] == batch_size and t[1] == seq]
                orig_mem_entries = [m for m in results["original"]["memory"] if m[0] == batch_size and m[1] == seq]
                hla_mem_entries = [m for m in results["hla"]["memory"] if m[0] == batch_size and m[1] == seq]
                
                if orig_entries:
                    times = [entry[2] for entry in orig_entries]
                    stds = [entry[3] for entry in orig_entries]
                    orig_time.append(np.median(times)*1000)
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Batch Size = {batch_size}")
            
            ax1.errorbar(seq_lengths, orig_time, yerr=orig_std, label="Original RoPE", marker='o', capsize=5)
            ax1.errorbar(seq_lengths, hla_time, yerr=hla_std, label="HLA RoPE", marker='o', capsize=5)
            ax1.set_xlabel("Sequence Length")
            ax1.set_ylabel("Execution Time (ms)")
            ax1.set_title("RoPE Execution Time (Median)")
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(seq_lengths, orig_mem, label="Original RoPE", marker='o')
            ax2.plot(seq_lengths, hla_mem, label="HLA RoPE", marker='o')
            ax2.set_xlabel("Sequence Length")
            ax2.set_ylabel("Memory Usage (MB)")
            ax2.set_title("RoPE Memory Usage (Median)")
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"aggregated_rope_benchmark_results_batch_{batch_size}.png")
            plt.show()

if __name__ == "__main__":
    # Set larger default tensor type for better GPU utilization
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    
    # Set environment variables for better GPU performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable asynchronous CUDA execution
    
    benchmark = RoPEBenchmark(model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer")
    
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Use fewer sessions but larger batch sizes and models
    sessions = 20
    print(f"\n=== Running {sessions} Benchmark Sessions ===")
    aggregated_results = benchmark.run_multiple_sessions(
        sessions=sessions,
        batch_sizes=[1],  # Increased batch sizes for better GPU utilization
        seq_lengths=[512, 1024, 2048],
        head_dim=64,  # Increased head dimension
        num_heads=6,
        trials=20,  # Reduced trials to avoid OOM
        inner_loops=50
    )
    
    print("\nAggregated Benchmark Results:")
    for method in ["original", "hla"]:
        for tup in aggregated_results[method]["time"]:
            print(f"{method} - Batch: {tup[0]}, Seq: {tup[1]}, Median Time: {tup[2]*1000:.2f} ms ± {tup[3]*1000:.2f} ms")
        for tup in aggregated_results[method]["memory"]:
            print(f"{method} - Batch: {tup[0]}, Seq: {tup[1]}, Memory Usage: {tup[2]:.2f} MB")
    
    benchmark.plot_results(aggregated_results)
    
    print("\n=== Running Detailed Profiling (batch_size=1, seq_len=2048) ===")
    profiling_metrics = benchmark.profile_rope_comparison_multi(batch_size=1, seq_len=2048, iterations=5)
    benchmark.plot_profiling_comparison(profiling_metrics)
    
    print("\nBenchmarking and profiling completed. Check the generated PNG files for graphs.")
    

# === GPU Information ===
# CUDA Device: NVIDIA A100-SXM4-40GB
# CUDA Version: 12.4
# CUDA Capability: (8, 0)
# Total GPU Memory: 39.56 GB

# === Running 20 Benchmark Sessions ===

# === Benchmark Session 1/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 2/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 29%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 3/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 4/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 5/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 6/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 7/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 8/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 9/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 10/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 11/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 38%
# Cooling GPU between sessions...

# === Benchmark Session 12/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 13/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 14/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 35%
# Cooling GPU between sessions...

# === Benchmark Session 15/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 16/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 17/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 18/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%
# Cooling GPU between sessions...

# === Benchmark Session 19/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 35%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 36%
# Cooling GPU between sessions...

# === Benchmark Session 20/20 ===

# Benchmarking with batch_size=1, seq_len=512
#   Original RoPE GPU memory: 8.63 MB
#   HLA RoPE GPU memory: 8.63 MB
# Speedup (HLA vs Original): 0.03x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=1024
#   Original RoPE GPU memory: 9.13 MB
#   HLA RoPE GPU memory: 9.13 MB
# Speedup (HLA vs Original): 0.02x
#   GPU utilization: 36%

# Benchmarking with batch_size=1, seq_len=2048
#   Original RoPE GPU memory: 10.14 MB
#   HLA RoPE GPU memory: 10.14 MB
# Speedup (HLA vs Original): 0.01x
#   GPU utilization: 37%

# Aggregated Benchmark Results:
# original - Batch: 1, Seq: 512, Median Time: 0.36 ms ± 0.00 ms
# original - Batch: 1, Seq: 1024, Median Time: 0.36 ms ± 0.01 ms
# original - Batch: 1, Seq: 2048, Median Time: 0.37 ms ± 0.01 ms
# original - Batch: 1, Seq: 512, Memory Usage: 0.13 MB
# original - Batch: 1, Seq: 1024, Memory Usage: 0.12 MB
# original - Batch: 1, Seq: 2048, Memory Usage: 0.25 MB
# hla - Batch: 1, Seq: 512, Median Time: 11.85 ms ± 0.05 ms
# hla - Batch: 1, Seq: 1024, Median Time: 23.16 ms ± 0.06 ms
# hla - Batch: 1, Seq: 2048, Median Time: 45.40 ms ± 0.09 ms
# hla - Batch: 1, Seq: 512, Memory Usage: 0.00 MB
# hla - Batch: 1, Seq: 1024, Memory Usage: 0.00 MB
# hla - Batch: 1, Seq: 2048, Memory Usage: 0.00 MB


# === Running Detailed Profiling (batch_size=1, seq_len=2048) ===
# Initial GPU memory usage: 16.14 MB
# Max GPU memory allocated: 16.14 MB

# Profiling iteration 1/5
# Original RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                           original_rope         0.00%       0.000us         0.00%       0.000us       0.000us      82.662ms      1350.88%      82.662ms      82.662ms           0 b           0 b           0 b           0 b             1            --  
#                                           original_rope        39.86%      38.712ms        99.99%      97.121ms      97.121ms       0.000us         0.00%       6.119ms       6.119ms           0 b           0 b     512.00 Kb    -325.48 Mb             1            --  
#                                               aten::sum         6.00%       5.824ms         9.20%       8.940ms      44.698us       2.099ms        34.31%       2.099ms      10.497us           0 b           0 b     100.00 Kb     100.00 Kb           200            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.827ms        29.85%       1.827ms       9.134us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         3.68%       3.573ms        16.79%      16.310ms      23.300us       0.000us         0.00%       1.047ms       1.496us           0 b           0 b      50.78 Mb           0 b           700            --  
#                                          aten::_to_copy         4.92%       4.779ms        13.11%      12.737ms      42.455us       0.000us         0.00%       1.047ms       3.491us           0 b           0 b      50.78 Mb           0 b           300            --  
#                                             aten::copy_         2.83%       2.751ms         5.81%       5.639ms      18.798us       1.047ms        17.11%       1.047ms       3.491us           0 b           0 b           0 b           0 b           300            --  
#                                               aten::cat         2.28%       2.216ms         3.40%       3.299ms      32.995us     748.984us        12.24%     748.984us       7.490us           0 b           0 b      50.00 Mb      50.00 Mb           100            --  
# void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     748.984us        12.24%     748.984us       7.490us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         2.91%       2.823ms         4.48%       4.348ms      21.741us     589.490us         9.63%     589.490us       2.947us           0 b           0 b     100.00 Mb     100.00 Mb           200  26214400.000  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 97.130ms
# Self CUDA time total: 6.119ms

#   Original RoPE peak memory: 19.40 MB
#   Original RoPE CUDA time (from profiler): 0.0000 μs
# HLA RoPE measured CUDA elapsed time (manual timing): 311.2826 ms
# HLA RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                hla_rope         0.00%       0.000us         0.00%       0.000us       0.000us      90.533ms      1659.91%      90.533ms      90.533ms           0 b           0 b           0 b           0 b             1            --  
#                                                hla_rope        41.29%      38.642ms        99.98%      93.560ms      93.560ms       0.000us         0.00%       5.454ms       5.454ms           0 b           0 b           0 b    -125.83 Mb             1            --  
#                                             aten::copy_         5.48%       5.129ms        10.26%       9.600ms      19.201us       2.198ms        40.30%       2.198ms       4.396us           0 b           0 b           0 b           0 b           500            --  
# void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.723ms        31.59%       1.723ms       4.307us           0 b           0 b           0 b           0 b           400            --  
#                                               aten::sum         3.39%       3.175ms         5.30%       4.957ms      49.570us       1.057ms        19.39%       1.057ms      10.575us           0 b           0 b      50.00 Kb      50.00 Kb           100            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     921.218us        16.89%     921.218us       9.212us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.09%       2.894ms         4.76%       4.457ms      22.283us     511.833us         9.38%     511.833us       2.559us           0 b           0 b      25.00 Mb      25.00 Mb           200         6.554  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     511.833us         9.38%     511.833us       2.559us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.39%     363.928us         5.21%       4.875ms       9.750us       0.000us         0.00%     475.044us       0.950us           0 b           0 b     800.00 Kb           0 b           500            --  
#                                          aten::_to_copy         0.95%     891.246us         4.82%       4.511ms      45.113us       0.000us         0.00%     475.044us       4.750us           0 b           0 b     800.00 Kb           0 b           100            --  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 93.575ms
# Self CUDA time total: 5.454ms

#   HLA RoPE peak memory: 18.15 MB
#   HLA RoPE CUDA time (from profiler): 0.0000 μs

# Profiling iteration 2/5
# Original RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                           original_rope         0.00%       0.000us         0.00%       0.000us       0.000us      82.078ms      1347.27%      82.078ms      82.078ms           0 b           0 b           0 b           0 b             1            --  
#                                           original_rope        37.44%      31.873ms        99.99%      85.127ms      85.127ms       0.000us         0.00%       6.092ms       6.092ms           0 b           0 b           0 b    -325.98 Mb             1            --  
#                                               aten::sum         6.64%       5.652ms        10.27%       8.739ms      43.697us       2.047ms        33.61%       2.047ms      10.237us           0 b           0 b     100.00 Kb     100.00 Kb           200            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.759ms        28.87%       1.759ms       8.794us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.99%     845.038us        13.09%      11.142ms      15.917us       0.000us         0.00%       1.047ms       1.496us           0 b           0 b      50.78 Mb           0 b           700            --  
#                                          aten::_to_copy         2.74%       2.330ms        12.09%      10.297ms      34.322us       0.000us         0.00%       1.047ms       3.491us           0 b           0 b      50.78 Mb           0 b           300            --  
#                                             aten::copy_         3.26%       2.773ms         6.66%       5.667ms      18.890us       1.047ms        17.19%       1.047ms       3.491us           0 b           0 b           0 b           0 b           300            --  
#                                               aten::cat         2.63%       2.236ms         3.89%       3.312ms      33.120us     748.632us        12.29%     748.632us       7.486us           0 b           0 b      50.00 Mb      50.00 Mb           100            --  
# void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     748.632us        12.29%     748.632us       7.486us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.39%       2.885ms         5.20%       4.426ms      22.132us     589.594us         9.68%     589.594us       2.948us           0 b           0 b     100.00 Mb     100.00 Mb           200  26214400.000  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 85.133ms
# Self CUDA time total: 6.092ms

#   Original RoPE peak memory: 19.40 MB
#   Original RoPE CUDA time (from profiler): 0.0000 μs
# HLA RoPE measured CUDA elapsed time (manual timing): 290.4932 ms
# HLA RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                hla_rope         0.00%       0.000us         0.00%       0.000us       0.000us      88.602ms      1630.23%      88.602ms      88.602ms           0 b           0 b           0 b           0 b             1            --  
#                                                hla_rope        37.09%      33.534ms        99.99%      90.405ms      90.405ms       0.000us         0.00%       5.435ms       5.435ms           0 b           0 b           0 b    -125.83 Mb             1            --  
#                                             aten::copy_         5.80%       5.247ms        11.33%      10.243ms      20.487us       2.186ms        40.22%       2.186ms       4.372us           0 b           0 b           0 b           0 b           500            --  
# void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.712ms        31.49%       1.712ms       4.279us           0 b           0 b           0 b           0 b           400            --  
#                                               aten::sum         3.63%       3.278ms         5.69%       5.146ms      51.464us       1.056ms        19.43%       1.056ms      10.561us           0 b           0 b      50.00 Kb      50.00 Kb           100            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     919.353us        16.92%     919.353us       9.194us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.31%       2.995ms         5.06%       4.572ms      22.861us     513.090us         9.44%     513.090us       2.565us           0 b           0 b      25.00 Mb      25.00 Mb           200         6.554  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     513.090us         9.44%     513.090us       2.565us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.42%     376.732us         6.07%       5.490ms      10.980us       0.000us         0.00%     474.430us       0.949us           0 b           0 b     800.00 Kb           0 b           500            --  
#                                          aten::_to_copy         1.02%     919.956us         5.66%       5.113ms      51.132us       0.000us         0.00%     474.430us       4.744us           0 b           0 b     800.00 Kb           0 b           100            --  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 90.410ms
# Self CUDA time total: 5.435ms

#   HLA RoPE peak memory: 18.15 MB
#   HLA RoPE CUDA time (from profiler): 0.0000 μs

# Profiling iteration 3/5
# Original RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                           original_rope         0.00%       0.000us         0.00%       0.000us       0.000us      83.473ms      1366.70%      83.473ms      83.473ms           0 b           0 b           0 b           0 b             1            --  
#                                           original_rope        35.37%      30.159ms        99.99%      85.258ms      85.258ms       0.000us         0.00%       6.108ms       6.108ms           0 b           0 b           0 b    -325.98 Mb             1            --  
#                                               aten::sum         6.86%       5.852ms        10.55%       8.999ms      44.996us       2.094ms        34.29%       2.094ms      10.470us           0 b           0 b     100.00 Kb     100.00 Kb           200            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.821ms        29.81%       1.821ms       9.104us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         1.03%     877.908us        14.20%      12.106ms      17.294us       0.000us         0.00%       1.045ms       1.493us           0 b           0 b      50.78 Mb           0 b           700            --  
#                                          aten::_to_copy         2.88%       2.456ms        13.17%      11.228ms      37.426us       0.000us         0.00%       1.045ms       3.483us           0 b           0 b      50.78 Mb           0 b           300            --  
#                                             aten::copy_         3.33%       2.836ms         7.42%       6.328ms      21.093us       1.045ms        17.11%       1.045ms       3.483us           0 b           0 b           0 b           0 b           300            --  
#                                               aten::cat         2.74%       2.332ms         4.03%       3.434ms      34.343us     747.803us        12.24%     747.803us       7.478us           0 b           0 b      50.00 Mb      50.00 Mb           100            --  
# void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     747.803us        12.24%     747.803us       7.478us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.42%       2.917ms         5.24%       4.470ms      22.352us     588.244us         9.63%     588.244us       2.941us           0 b           0 b     100.00 Mb     100.00 Mb           200  26214400.000  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 85.264ms
# Self CUDA time total: 6.108ms

#   Original RoPE peak memory: 19.40 MB
#   Original RoPE CUDA time (from profiler): 0.0000 μs
# HLA RoPE measured CUDA elapsed time (manual timing): 285.1781 ms
# HLA RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                hla_rope         0.00%       0.000us         0.00%       0.000us       0.000us      86.011ms      1575.43%      86.011ms      86.011ms           0 b           0 b           0 b           0 b             1            --  
#                                                hla_rope        36.93%      32.415ms       100.00%      87.764ms      87.764ms       0.000us         0.00%       5.460ms       5.460ms           0 b           0 b           0 b    -125.83 Mb             1            --  
#                                             aten::copy_         5.84%       5.126ms        11.48%      10.075ms      20.151us       2.204ms        40.37%       2.204ms       4.408us           0 b           0 b           0 b           0 b           500            --  
# void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.729ms        31.66%       1.729ms       4.322us           0 b           0 b           0 b           0 b           400            --  
#                                               aten::sum         3.61%       3.169ms         5.64%       4.949ms      49.489us       1.057ms        19.35%       1.057ms      10.566us           0 b           0 b      50.00 Kb      50.00 Kb           100            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     919.964us        16.85%     919.964us       9.200us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.33%       2.927ms         5.10%       4.474ms      22.368us     513.788us         9.41%     513.788us       2.569us           0 b           0 b      25.00 Mb      25.00 Mb           200         6.554  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     513.788us         9.41%     513.788us       2.569us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.40%     348.397us         6.06%       5.321ms      10.642us       0.000us         0.00%     475.301us       0.951us           0 b           0 b     800.00 Kb           0 b           500            --  
#                                          aten::_to_copy         1.03%     905.043us         5.67%       4.973ms      49.728us       0.000us         0.00%     475.301us       4.753us           0 b           0 b     800.00 Kb           0 b           100            --  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 87.768ms
# Self CUDA time total: 5.460ms

#   HLA RoPE peak memory: 18.15 MB
#   HLA RoPE CUDA time (from profiler): 0.0000 μs

# Profiling iteration 4/5
# Original RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                           original_rope         0.00%       0.000us         0.00%       0.000us       0.000us      83.550ms      1368.20%      83.550ms      83.550ms           0 b           0 b           0 b           0 b             1            --  
#                                           original_rope        35.20%      30.040ms        99.99%      85.347ms      85.347ms       0.000us         0.00%       6.107ms       6.107ms           0 b           0 b           0 b    -325.98 Mb             1            --  
#                                               aten::sum         6.80%       5.804ms        10.48%       8.944ms      44.720us       2.096ms        34.32%       2.096ms      10.480us           0 b           0 b     100.00 Kb     100.00 Kb           200            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.822ms        29.84%       1.822ms       9.110us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         1.00%     851.129us        14.22%      12.137ms      17.339us       0.000us         0.00%       1.044ms       1.491us           0 b           0 b      50.78 Mb           0 b           700            --  
#                                          aten::_to_copy         2.88%       2.462ms        13.22%      11.286ms      37.620us       0.000us         0.00%       1.044ms       3.479us           0 b           0 b      50.78 Mb           0 b           300            --  
#                                             aten::copy_         3.30%       2.814ms         7.45%       6.360ms      21.199us       1.044ms        17.09%       1.044ms       3.479us           0 b           0 b           0 b           0 b           300            --  
#                                               aten::cat         2.68%       2.292ms         4.04%       3.447ms      34.472us     747.000us        12.23%     747.000us       7.470us           0 b           0 b      50.00 Mb      50.00 Mb           100            --  
# void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     747.000us        12.23%     747.000us       7.470us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.41%       2.915ms         5.26%       4.486ms      22.429us     588.438us         9.64%     588.438us       2.942us           0 b           0 b     100.00 Mb     100.00 Mb           200  26214400.000  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 85.353ms
# Self CUDA time total: 6.107ms

#   Original RoPE peak memory: 19.40 MB
#   Original RoPE CUDA time (from profiler): 0.0000 μs
# HLA RoPE measured CUDA elapsed time (manual timing): 302.9495 ms
# HLA RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                hla_rope         0.00%       0.000us         0.00%       0.000us       0.000us      89.606ms      1652.78%      89.606ms      89.606ms           0 b           0 b           0 b           0 b             1            --  
#                                                hla_rope        37.15%      33.956ms        99.99%      91.408ms      91.408ms       0.000us         0.00%       5.421ms       5.421ms           0 b           0 b           0 b    -125.83 Mb             1            --  
#                                             aten::copy_         5.72%       5.231ms        11.35%      10.375ms      20.749us       2.194ms        40.48%       2.194ms       4.389us           0 b           0 b           0 b           0 b           500            --  
# void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.720ms        31.73%       1.720ms       4.300us           0 b           0 b           0 b           0 b           400            --  
#                                               aten::sum         3.53%       3.227ms         5.53%       5.056ms      50.562us       1.031ms        19.01%       1.031ms      10.308us           0 b           0 b      50.00 Kb      50.00 Kb           100            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     883.844us        16.30%     883.844us       8.838us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.23%       2.954ms         4.93%       4.506ms      22.529us     511.676us         9.44%     511.676us       2.558us           0 b           0 b      25.00 Mb      25.00 Mb           200         6.554  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     511.676us         9.44%     511.676us       2.558us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.40%     367.462us         6.01%       5.492ms      10.984us       0.000us         0.00%     474.459us       0.949us           0 b           0 b     800.00 Kb           0 b           500            --  
#                                          aten::_to_copy         1.01%     927.140us         5.61%       5.125ms      51.248us       0.000us         0.00%     474.459us       4.745us           0 b           0 b     800.00 Kb           0 b           100            --  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 91.414ms
# Self CUDA time total: 5.421ms

#   HLA RoPE peak memory: 18.15 MB
#   HLA RoPE CUDA time (from profiler): 0.0000 μs

# Profiling iteration 5/5
# Original RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls   Total FLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                           original_rope         0.00%       0.000us         0.00%       0.000us       0.000us      82.032ms      1357.34%      82.032ms      82.032ms           0 b           0 b           0 b           0 b             1            --  
#                                           original_rope        35.21%      29.498ms        99.99%      83.766ms      83.766ms       0.000us         0.00%       6.044ms       6.044ms           0 b           0 b           0 b    -325.98 Mb             1            --  
#                                               aten::sum         6.76%       5.664ms        10.46%       8.764ms      43.819us       1.986ms        32.87%       1.986ms       9.931us           0 b           0 b     100.00 Kb     100.00 Kb           200            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.708ms        28.26%       1.708ms       8.539us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         1.07%     898.282us        14.23%      11.923ms      17.033us       0.000us         0.00%       1.045ms       1.493us           0 b           0 b      50.78 Mb           0 b           700            --  
#                                          aten::_to_copy         2.83%       2.372ms        13.16%      11.025ms      36.750us       0.000us         0.00%       1.045ms       3.484us           0 b           0 b      50.78 Mb           0 b           300            --  
#                                             aten::copy_         3.29%       2.758ms         7.48%       6.267ms      20.889us       1.045ms        17.30%       1.045ms       3.484us           0 b           0 b           0 b           0 b           300            --  
#                                               aten::cat         2.67%       2.235ms         3.95%       3.310ms      33.097us     748.919us        12.39%     748.919us       7.489us           0 b           0 b      50.00 Mb      50.00 Mb           100            --  
# void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     748.919us        12.39%     748.919us       7.489us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.45%       2.887ms         5.31%       4.446ms      22.231us     588.221us         9.73%     588.221us       2.941us           0 b           0 b     100.00 Mb     100.00 Mb           200  26214400.000  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 83.771ms
# Self CUDA time total: 6.044ms

#   Original RoPE peak memory: 19.40 MB
#   Original RoPE CUDA time (from profiler): 0.0000 μs
# HLA RoPE measured CUDA elapsed time (manual timing): 303.8152 ms
# HLA RoPE Profile Events:
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                hla_rope         0.00%       0.000us         0.00%       0.000us       0.000us      86.684ms      1590.04%      86.684ms      86.684ms           0 b           0 b           0 b           0 b             1            --  
#                                                hla_rope        37.39%      33.071ms        99.99%      88.443ms      88.443ms       0.000us         0.00%       5.452ms       5.452ms           0 b           0 b           0 b    -125.83 Mb             1            --  
#                                             aten::copy_         5.67%       5.017ms        11.25%       9.949ms      19.897us       2.192ms        40.21%       2.192ms       4.385us           0 b           0 b           0 b           0 b           500            --  
# void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.718ms        31.51%       1.718ms       4.294us           0 b           0 b           0 b           0 b           400            --  
#                                               aten::sum         3.49%       3.084ms         5.48%       4.846ms      48.456us       1.072ms        19.66%       1.072ms      10.715us           0 b           0 b      50.00 Kb      50.00 Kb           100            --  
# void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     922.865us        16.93%     922.865us       9.229us           0 b           0 b           0 b           0 b           100            --  
#                                               aten::mul         3.23%       2.861ms         4.97%       4.398ms      21.989us     511.040us         9.37%     511.040us       2.555us           0 b           0 b      25.00 Mb      25.00 Mb           200         6.554  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     511.040us         9.37%     511.040us       2.555us           0 b           0 b           0 b           0 b           200            --  
#                                                aten::to         0.41%     362.650us         5.98%       5.286ms      10.572us       0.000us         0.00%     474.675us       0.949us           0 b           0 b     800.00 Kb           0 b           500            --  
#                                          aten::_to_copy         1.02%     900.969us         5.57%       4.923ms      49.231us       0.000us         0.00%     474.675us       4.747us           0 b           0 b     800.00 Kb           0 b           100            --  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 88.448ms
# Self CUDA time total: 5.452ms

#   HLA RoPE peak memory: 18.15 MB
#   HLA RoPE CUDA time (from profiler): 0.0000 μs

# CUDA Operations Profile Summary:
# Original RoPE CUDA Time (from profiler): 0.0000 μs
# HLA RoPE CUDA Time (from profiler): 0.0000 μs
# HLA RoPE CUDA Time (manual timing): 303.8152 ms


