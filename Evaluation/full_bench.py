import torch
import time
import gc
import numpy as np
import psutil
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.profiler import profile, record_function
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from modeling_llama_HLA import LlamaForCausalLM as LlamaForCausalLM_HLA

class LlamaBenchmark:
    def __init__(self, model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer", enable_profiling=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.process = psutil.Process(os.getpid())
        self.enable_profiling = enable_profiling

        # Create output directory
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def _clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _time_and_memory_function(self, func, *args, trials=10, **kwargs):
        times = []
        peak_memories = []
        result = None
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                result = func(*args, **kwargs)

        for _ in range(trials):
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    result = func(*args, **kwargs)
                end_event.record()
                torch.cuda.synchronize()
                trial_time = start_event.elapsed_time(end_event) / 1000.0
                trial_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                gc.collect()
                start_mem = self.process.memory_info().rss
                start_time = time.time()
                with torch.no_grad():
                    result = func(*args, **kwargs)
                trial_time = time.time() - start_time
                end_mem = self.process.memory_info().rss
                trial_peak = (end_mem - start_mem) / (1024 ** 2)
            times.append(trial_time)
            peak_memories.append(trial_peak)
        return np.mean(times), np.std(times), np.median(times), np.mean(peak_memories), result

    def load_models(self):
        print(f"Loading original Llama model from {self.model_name}...")
        self._clear_gpu_memory()
        self.original_model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.original_model.eval()

        print(f"Loading HLA Llama model from {self.model_name}...")
        self._clear_gpu_memory()
        self.hla_model = LlamaForCausalLM_HLA.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.hla_model.eval()

    def benchmark_inference(self, input_texts, seq_lengths=[128, 256, 512, 1024, 2048], gen_lengths=[20, 50, 100], trials=10):
        if not hasattr(self, 'original_model') or not hasattr(self, 'hla_model'):
            self.load_models()

        results = []
        for input_text in input_texts:
            for seq_length in seq_lengths:
                for gen_length in gen_lengths:
                    print(f"\nBenchmarking inference with seq_length={seq_length}, gen_length={gen_length}")
                    print(f"Input text: {input_text[:50]}{'...' if len(input_text) > 50 else ''}")

                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=seq_length,
                        return_attention_mask=True
                    ).to(self.device)

                    generation_kwargs = {
                        "max_length": seq_length + gen_length,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_k": 50,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id
                    }

                    # Benchmark original model
                    self._clear_gpu_memory()
                    orig_time, orig_std, orig_median, orig_peak_mem, orig_output = self._time_and_memory_function(
                        self.original_model.generate,
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        trials=trials,
                        **generation_kwargs
                    )

                    # Optional profiling (run once if enabled)
                    if self.enable_profiling:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as orig_prof:
                            with record_function("original_model_generate"):
                                self.original_model.generate(
                                    inputs.input_ids,
                                    attention_mask=inputs.attention_mask,
                                    **generation_kwargs
                                )

                    orig_decoded = self.tokenizer.decode(orig_output[0], skip_special_tokens=True)
                    print(f"Original Llama: Mean Time: {orig_time*1000:.2f} ms (Median: {orig_median*1000:.2f} ms), Std: {orig_std*1000:.2f} ms, Peak Memory: {orig_peak_mem:.2f} MB")

                    # Benchmark HLA model
                    self._clear_gpu_memory()
                    hla_time, hla_std, hla_median, hla_peak_mem, hla_output = self._time_and_memory_function(
                        self.hla_model.generate,
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        trials=trials,
                        **generation_kwargs
                    )

                    if self.enable_profiling:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as hla_prof:
                            with record_function("hla_model_generate"):
                                self.hla_model.generate(
                                    inputs.input_ids,
                                    attention_mask=inputs.attention_mask,
                                    **generation_kwargs
                                )

                    hla_decoded = self.tokenizer.decode(hla_output[0], skip_special_tokens=True)
                    print(f"HLA Llama: Mean Time: {hla_time*1000:.2f} ms (Median: {hla_median*1000:.2f} ms), Std: {hla_std*1000:.2f} ms, Peak Memory: {hla_peak_mem:.2f} MB")

                    speedup = orig_time / hla_time if hla_time > 0 else float('inf')
                    memory_reduction = (orig_peak_mem - hla_peak_mem) / orig_peak_mem * 100 if orig_peak_mem > 0 else 0
                    print(f"Speedup (HLA vs Original): {speedup:.2f}x")
                    print(f"Memory reduction: {memory_reduction:.2f}%")

                    bleu_score = self.compute_similarity(orig_decoded, hla_decoded)
                    print(f"Output similarity score: {bleu_score:.4f}")

                    results.append({
                        "input_length": len(inputs.input_ids[0]),
                        "gen_length": gen_length,
                        "orig_mean_time_ms": orig_time * 1000,
                        "orig_median_time_ms": orig_median * 1000,
                        "hla_mean_time_ms": hla_time * 1000,
                        "hla_median_time_ms": hla_median * 1000,
                        "orig_peak_memory_mb": orig_peak_mem,
                        "hla_peak_memory_mb": hla_peak_mem,
                        "speedup": speedup,
                        "memory_reduction": memory_reduction,
                        "output_similarity": bleu_score
                    })

                    print("\nSample output comparison:")
                    print(f"Original: {orig_decoded[:100]}...")
                    print(f"HLA: {hla_decoded[:100]}...")

        df = pd.DataFrame(results)
        return df

    def benchmark_training(self, batch_sizes=[1, 2, 4], seq_lengths=[128, 256, 512, 1024], trials=10):
        if not hasattr(self, 'original_model') or not hasattr(self, 'hla_model'):
            self.load_models()

        self.original_model.train()
        self.hla_model.train()
        results = []

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                print(f"\nBenchmarking training with batch_size={batch_size}, seq_length={seq_length}")
                input_ids = torch.randint(
                    0, self.tokenizer.vocab_size,
                    (batch_size, seq_length),
                    dtype=torch.long, device=self.device
                )
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()

                self._clear_gpu_memory()
                def train_step_original():
                    self.original_model.zero_grad()
                    
                    # Ensure model parameters require gradients
                    for param in self.original_model.parameters():
                        param.requires_grad = True
                        
                    outputs = self.original_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    
                    # Check if we're in a no_grad context before calling backward
                    if torch.is_grad_enabled():
                        loss.backward()
                        
                    return loss.item()
                orig_time, orig_std, orig_median, orig_peak_mem, orig_loss = self._time_and_memory_function(train_step_original, trials=trials)
                print(f"Original Llama: Mean Time: {orig_time*1000:.2f} ms (Median: {orig_median*1000:.2f} ms), Std: {orig_std*1000:.2f} ms, Peak Memory: {orig_peak_mem:.2f} MB, Loss: {orig_loss:.4f}")

                self._clear_gpu_memory()
                def train_step_hla():
                    self.hla_model.zero_grad()
                    
                    # Ensure model parameters require gradients
                    for param in self.hla_model.parameters():
                        param.requires_grad = True
                        
                    outputs = self.hla_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    
                    # Check if we're in a no_grad context before calling backward
                    if torch.is_grad_enabled():
                        loss.backward()
                        
                    return loss.item()

                hla_time, hla_std, hla_median, hla_peak_mem, hla_loss = self._time_and_memory_function(train_step_hla, trials=trials)
                print(f"HLA Llama: Mean Time: {hla_time*1000:.2f} ms (Median: {hla_median*1000:.2f} ms), Std: {hla_std*1000:.2f} ms, Peak Memory: {hla_peak_mem:.2f} MB, Loss: {hla_loss:.4f}")

                speedup = orig_time / hla_time if hla_time > 0 else float('inf')
                memory_reduction = (orig_peak_mem - hla_peak_mem) / orig_peak_mem * 100 if orig_peak_mem > 0 else 0
                loss_diff = abs(orig_loss - hla_loss)
                print(f"Speedup (HLA vs Original): {speedup:.2f}x")
                print(f"Memory reduction: {memory_reduction:.2f}%")
                print(f"Loss difference: {loss_diff:.6f}")

                results.append({
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "orig_mean_time_ms": orig_time * 1000,
                    "orig_median_time_ms": orig_median * 1000,
                    "hla_mean_time_ms": hla_time * 1000,
                    "hla_median_time_ms": hla_median * 1000,
                    "orig_peak_memory_mb": orig_peak_mem,
                    "hla_peak_memory_mb": hla_peak_mem,
                    "speedup": speedup,
                    "memory_reduction": memory_reduction,
                    "orig_loss": orig_loss,
                    "hla_loss": hla_loss,
                    "loss_diff": loss_diff
                })

        self.original_model.eval()
        self.hla_model.eval()
        df = pd.DataFrame(results)
        df.to_csv(f"{self.output_dir}/training_benchmark_results.csv", index=False)
        return df

    def benchmark_scalability(self, seq_lengths=[512, 1024, 2048, 4096, 8192], trials=10):
        results = []
        for seq_length in seq_lengths:
            print(f"\nBenchmarking scalability with seq_length={seq_length}")
            try:
                input_ids = torch.randint(
                    0, self.tokenizer.vocab_size,
                    (1, seq_length),
                    dtype=torch.long, device=self.device
                )
                attention_mask = torch.ones_like(input_ids)
                self._clear_gpu_memory()
                try:
                    orig_time, orig_std, orig_median, _, _ = self._time_and_memory_function(
                        self.original_model.forward,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        trials=trials
                    )
                    orig_success = True
                    print(f"Original Llama: Mean Time: {orig_time*1000:.2f} ms (Median: {orig_median*1000:.2f} ms), Std: {orig_std*1000:.2f} ms")
                except RuntimeError as e:
                    print(f"Original model failed at seq_length={seq_length}: {str(e)}")
                    orig_time, orig_std = float('inf'), 0
                    orig_success = False

                self._clear_gpu_memory()
                try:
                    hla_time, hla_std, hla_median, _, _ = self._time_and_memory_function(
                        self.hla_model.forward,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        trials=trials
                    )
                    hla_success = True
                    print(f"HLA Llama: Mean Time: {hla_time*1000:.2f} ms (Median: {hla_median*1000:.2f} ms), Std: {hla_std*1000:.2f} ms")
                except RuntimeError as e:
                    print(f"HLA model failed at seq_length={seq_length}: {str(e)}")
                    hla_time, hla_std = float('inf'), 0
                    hla_success = False

                speedup = orig_time / hla_time if hla_time > 0 else float('nan')
                if orig_success and hla_success:
                    print(f"Speedup (HLA vs Original): {speedup:.2f}x")
                else:
                    print("Speedup could not be computed due to failure.")

                results.append({
                    "seq_length": seq_length,
                    "orig_mean_time_ms": orig_time * 1000 if orig_success else float('nan'),
                    "hla_mean_time_ms": hla_time * 1000 if hla_success else float('nan'),
                    "speedup": speedup,
                    "orig_success": orig_success,
                    "hla_success": hla_success
                })
            except Exception as e:
                print(f"Error during scalability benchmark at seq_length={seq_length}: {str(e)}")
                results.append({
                    "seq_length": seq_length,
                    "orig_mean_time_ms": float('nan'),
                    "hla_mean_time_ms": float('nan'),
                    "speedup": float('nan'),
                    "orig_success": False,
                    "hla_success": False
                })
        df = pd.DataFrame(results)
        df.to_csv(f"{self.output_dir}/scalability_benchmark_results.csv", index=False)
        return df

    def plot_all_results(self, inference_df=None, training_df=None, scalability_df=None):
        plt.style.use('ggplot')
        fig_size = (15, 12)
        fig, axes = plt.subplots(3, 2, figsize=fig_size)

        if inference_df is not None:
            sns.lineplot(
                data=inference_df,
                x="input_length",
                y="speedup",
                marker='o',
                ax=axes[0, 0]
            )
            axes[0, 0].set_title("Inference Speedup by Sequence Length")
            axes[0, 0].set_xlabel("Input Length (tokens)")
            axes[0, 0].set_ylabel("Speedup (x)")
            axes[0, 0].grid(True)

            sns.lineplot(
                data=inference_df,
                x="input_length",
                y="memory_reduction",
                marker='o',
                ax=axes[0, 1]
            )
            axes[0, 1].set_title("Memory Reduction by Sequence Length")
            axes[0, 1].set_xlabel("Input Length (tokens)")
            axes[0, 1].set_ylabel("Memory Reduction (%)")
            axes[0, 1].grid(True)

        if training_df is not None:
            training_by_seq = training_df.groupby("seq_length").mean().reset_index()

            sns.lineplot(
                data=training_by_seq,
                x="seq_length",
                y="speedup",
                marker='o',
                ax=axes[1, 0]
            )
            axes[1, 0].set_title("Training Speedup by Sequence Length")
            axes[1, 0].set_xlabel("Sequence Length (tokens)")
            axes[1, 0].set_ylabel("Speedup (x)")
            axes[1, 0].grid(True)

            sns.lineplot(
                data=training_by_seq,
                x="seq_length",
                y="memory_reduction",
                marker='o',
                ax=axes[1, 1]
            )
            axes[1, 1].set_title("Training Memory Reduction by Sequence Length")
            axes[1, 1].set_xlabel("Sequence Length (tokens)")
            axes[1, 1].set_ylabel("Memory Reduction (%)")
            axes[1, 1].grid(True)

        if scalability_df is not None:
            scalability_long = pd.melt(
                scalability_df,
                id_vars=['seq_length'],
                value_vars=['orig_mean_time_ms', 'hla_mean_time_ms'],
                var_name='model',
                value_name='time_ms'
            )
            scalability_long['model'] = scalability_long['model'].map({
                'orig_mean_time_ms': 'Original',
                'hla_mean_time_ms': 'HLA'
            })

            sns.lineplot(
                data=scalability_long,
                x="seq_length",
                y="time_ms",
                hue="model",
                marker='o',
                ax=axes[2, 0]
            )
            axes[2, 0].set_title("Execution Time vs Sequence Length")
            axes[2, 0].set_xlabel("Sequence Length (tokens)")
            axes[2, 0].set_ylabel("Time (ms)")
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(True)

            sns.lineplot(
                data=scalability_df,
                x="seq_length",
                y="speedup",
                marker='o',
                ax=axes[2, 1]
            )
            axes[2, 1].set_title("Scalability: Speedup by Sequence Length")
            axes[2, 1].set_xlabel("Sequence Length (tokens)")
            axes[2, 1].set_ylabel("Speedup (x)")
            axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/benchmark_summary.png", dpi=300)
        plt.show()

        summary_data = []
        if inference_df is not None:
            avg_inf_speedup = inference_df['speedup'].mean()
            max_inf_speedup = inference_df['speedup'].max()
            avg_inf_mem_reduction = inference_df['memory_reduction'].mean()
            summary_data.append({'Metric': 'Inference Average Speedup', 'Value': f"{avg_inf_speedup:.2f}x"})
            summary_data.append({'Metric': 'Inference Maximum Speedup', 'Value': f"{max_inf_speedup:.2f}x"})
            summary_data.append({'Metric': 'Inference Average Memory Reduction', 'Value': f"{avg_inf_mem_reduction:.2f}%"})
        if training_df is not None:
            avg_train_speedup = training_df['speedup'].mean()
            max_train_speedup = training_df['speedup'].max()
            avg_train_mem_reduction = training_df['memory_reduction'].mean()
            summary_data.append({'Metric': 'Training Average Speedup', 'Value': f"{avg_train_speedup:.2f}x"})
            summary_data.append({'Metric': 'Training Maximum Speedup', 'Value': f"{max_train_speedup:.2f}x"})
            summary_data.append({'Metric': 'Training Average Memory Reduction', 'Value': f"{avg_train_mem_reduction:.2f}%"})
        if scalability_df is not None:
            orig_max_seq = scalability_df[scalability_df['orig_success']]['seq_length'].max()
            hla_max_seq = scalability_df[scalability_df['hla_success']]['seq_length'].max()
            summary_data.append({'Metric': 'Original Max Successful Sequence Length', 'Value': f"{orig_max_seq}"})
            summary_data.append({'Metric': 'HLA Max Successful Sequence Length', 'Value': f"{hla_max_seq}"})

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/benchmark_summary.csv", index=False)
        return summary_df

    def compute_similarity(self, text1, text2):
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        try:
            reference = [word_tokenize(text1.lower())]
            candidate = word_tokenize(text2.lower())
            return sentence_bleu(reference, candidate)
        except:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 1.0

    def run_full_benchmark(self):
        self.load_models()
        input_texts = [
            "The future of AI is",
            "In a world where technology and human creativity intersect,",
            # "The quick brown fox jumps over the lazy dog. " * 10,
            # "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer"
        ]

        print("\n=== Running Inference Benchmark ===")
        inference_df = self.benchmark_inference(
            input_texts=input_texts,
            seq_lengths=[128, 256],
            gen_lengths=[10],
            trials=1
        )

        print("\n=== Running Training Benchmark ===")
        training_df = self.benchmark_training(
            batch_sizes=[1],
            seq_lengths=[128, 256],
            trials=10
        )

        print("\n=== Running Scalability Benchmark ===")
        scalability_df = self.benchmark_scalability(
            seq_lengths=[512, 1024],
            trials=10
        )

        print("\n=== Generating Summary and Plots ===")
        summary_df = self.plot_all_results(
            inference_df=inference_df,
            training_df=training_df,
            scalability_df=scalability_df
        )

        print("\n=== Benchmark Results Summary ===")
        print(summary_df.to_string(index=False))

        print(f"\nDetailed results and plots saved to {self.output_dir}/")
        return inference_df, training_df, scalability_df, summary_df

if __name__ == "__main__":
    benchmark = LlamaBenchmark(model_name="AICrossSim/clm-60m", tokenizer_name="HuggingFaceTB/cosmo2-tokenizer", enable_profiling=True)
    benchmark.run_full_benchmark()

# === Running Inference Benchmark ===

# Benchmarking inference with seq_length=128, gen_length=10
# Input text: The future of AI is
# Original Llama: Mean Time: 3632.05 ms (Median: 3632.05 ms), Std: 0.00 ms, Peak Memory: 336.65 MB
# HLA Llama: Mean Time: 33968.03 ms (Median: 33968.03 ms), Std: 0.00 ms, Peak Memory: 367.64 MB
# Speedup (HLA vs Original): 0.11x
# Memory reduction: -9.21%
# Output similarity score: 0.1143

# Sample output comparison:
# Original: The future of AI is to create a unique environment.
# The global economy is a 1990s climate that is a ...
# HLA: The future of AI is to bring the way to protect against the body mass of the front--------------stat...

# Benchmarking inference with seq_length=256, gen_length=10
# Input text: The future of AI is
# Original Llama: Mean Time: 8678.26 ms (Median: 8678.26 ms), Std: 0.00 ms, Peak Memory: 338.03 MB
# HLA Llama: Mean Time: 67079.44 ms (Median: 67079.44 ms), Std: 0.00 ms, Peak Memory: 367.65 MB
# Speedup (HLA vs Original): 0.13x
# Memory reduction: -8.76%
# Output similarity score: 0.1277

# Sample output comparison:
# Original: The future of AI is to be implemented by the United Nations, including the United Nations, the Unite...
# HLA: The future of AI is that this will continue to the sun as the central the back to the 'Tg between th...

# Benchmarking inference with seq_length=128, gen_length=10
# Input text: In a world where technology and human creativity i...
# Original Llama: Mean Time: 4562.43 ms (Median: 4562.43 ms), Std: 0.00 ms, Peak Memory: 336.66 MB
# HLA Llama: Mean Time: 32937.73 ms (Median: 32937.73 ms), Std: 0.00 ms, Peak Memory: 402.39 MB
# Speedup (HLA vs Original): 0.14x
# Memory reduction: -19.53%
# Output similarity score: 0.1846

# Sample output comparison:
# Original: In a world where technology and human creativity intersect, there are so many other things that we c...
# HLA: In a world where technology and human creativity intersect, and the edge from the central bank S' to...

# Benchmarking inference with seq_length=256, gen_length=10
# Input text: In a world where technology and human creativity i...
# Original Llama: Mean Time: 8445.40 ms (Median: 8445.40 ms), Std: 0.00 ms, Peak Memory: 338.03 MB
# HLA Llama: Mean Time: 66006.45 ms (Median: 66006.45 ms), Std: 0.00 ms, Peak Memory: 402.39 MB
# Speedup (HLA vs Original): 0.13x
# Memory reduction: -19.04%
# Output similarity score: 0.1228

# Sample output comparison:
# Original: In a world where technology and human creativity intersect, how to build a new generation of knowled...
# HLA: In a world where technology and human creativity intersect, ‘’s natural resources to improve your do...

# === Running Training Benchmark ===

# Benchmarking training with batch_size=1, seq_length=128
# Original Llama: Mean Time: 35.25 ms (Median: 35.23 ms), Std: 0.30 ms, Peak Memory: 394.27 MB, Loss: 13.7032
# HLA Llama: Mean Time: 272.39 ms (Median: 271.82 ms), Std: 1.54 ms, Peak Memory: 1216.19 MB, Loss: 13.2813
# Speedup (HLA vs Original): 0.13x
# Memory reduction: -208.47%
# Loss difference: 0.421843

# Benchmarking training with batch_size=1, seq_length=256
# Original Llama: Mean Time: 38.26 ms (Median: 38.27 ms), Std: 2.15 ms, Peak Memory: 455.74 MB, Loss: 13.2994
# HLA Llama: Mean Time: 325.91 ms (Median: 325.70 ms), Std: 1.13 ms, Peak Memory: 2099.35 MB, Loss: 13.1890
# Speedup (HLA vs Original): 0.12x
# Memory reduction: -360.64%
# Loss difference: 0.110443

# === Running Scalability Benchmark ===

# Benchmarking scalability with seq_length=512
# Original Llama: Mean Time: 35.40 ms (Median: 35.24 ms), Std: 0.96 ms
# HLA Llama: Mean Time: 431.36 ms (Median: 431.27 ms), Std: 0.56 ms
# Speedup (HLA vs Original): 0.08x

# Benchmarking scalability with seq_length=1024
# Original Llama: Mean Time: 33.39 ms (Median: 33.36 ms), Std: 0.39 ms
# HLA Llama: Mean Time: 650.44 ms (Median: 649.67 ms), Std: 2.94 ms
# Speedup (HLA vs Original): 0.05x

# === Benchmark Results Summary ===
#                                  Metric    Value
#               Inference Average Speedup    0.13x
#               Inference Maximum Speedup    0.14x
#      Inference Average Memory Reduction  -14.13%
#                Training Average Speedup    0.12x
#                Training Maximum Speedup    0.13x
#       Training Average Memory Reduction -284.56%
# Original Max Successful Sequence Length     1024
#      HLA Max Successful Sequence Length     1024