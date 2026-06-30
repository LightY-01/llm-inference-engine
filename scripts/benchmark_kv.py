import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from data import Tokenizer
from generate import generate_naive, generate_with_kvcache
from inference import load_baseline_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'gpt_checkpoint.pth' 

model, context_length = load_baseline_model(checkpoint_path, device)

tokenizer = Tokenizer("gpt2") 

prompt = "To be, or not to be"
print(f"\nPrompt: {prompt}")

seq_lens = [50, 100, 200, 500, 1000]
num_runs = 3
results_list = []

for seq_len in seq_lens:
    print(f"Sequence length {seq_len}: ")
    naive_times = []
    kvcache_times = []
    naive_tokens_per_sec = []
    kvcache_tokens_per_sec = []
    
    for run in range(num_runs):
        # Naive Generation
        start_time = time.time()
        output_naive = generate_naive(
            model=model, 
            prompt=prompt, 
            tokenizer=tokenizer, 
            max_new_tokens=seq_len, 
            context_length=context_length, 
            device=device
        )
        time_naive = time.time() - start_time
        naive_times.append(time_naive)
        naive_tokens_per_sec.append(seq_len / time_naive)
        
        # KV Cache Generation
        start_time = time.time()
        output_kv = generate_with_kvcache(
            model=model, 
            prompt=prompt, 
            tokenizer=tokenizer, 
            max_new_tokens=seq_len, 
            context_length=context_length, 
            device=device
        )
        time_kv = time.time() - start_time
        kvcache_times.append(time_kv)
        kvcache_tokens_per_sec.append(seq_len / time_kv)
        print(f"\tRun {run + 1}: Naive Time = {time_naive:.4f}s, KV Cache Time = {time_kv:.4f}s")

    avg_naive_time = sum(naive_times) / num_runs
    avg_kvcache_time = sum(kvcache_times) / num_runs
    avg_naive_tokens_per_sec = sum(naive_tokens_per_sec) / num_runs
    avg_kvcache_tokens_per_sec = sum(kvcache_tokens_per_sec) / num_runs
    print(f"Average Naive Time = {avg_naive_time:.4f}s, Average KV Cache Time = {avg_kvcache_time:.4f}s")
    print(f"Average Naive Tokens per Second = {avg_naive_tokens_per_sec:.4f}, Average KV Cache Tokens per Second = {avg_kvcache_tokens_per_sec:.4f}")

    results_list.append({"seq_len": seq_len, "method": "naive", "time": avg_naive_time, "tokens/sec": avg_naive_tokens_per_sec})
    results_list.append({"seq_len": seq_len, "method": "kvcache", "time": avg_kvcache_time, "tokens/sec": avg_kvcache_tokens_per_sec})

benchmark = pd.DataFrame(results_list)
benchmark.to_csv("results/benchmark_results.csv", index=False)

# Plot 1: Tokens per second comparison
plt.figure(figsize=(10, 6))
plt.plot(benchmark[benchmark["method"] == "naive"]["seq_len"], benchmark[benchmark["method"] == "naive"]["tokens/sec"], marker='o', label="Naive")
plt.plot(benchmark[benchmark["method"] == "kvcache"]["seq_len"], benchmark[benchmark["method"] == "kvcache"]["tokens/sec"], marker='s', label="KV Cache")
plt.title("Tokens per Second vs. Sequence Length")
plt.xlabel("Sequence Length")
plt.ylabel("Tokens per Second")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("results/benchmark_tokens_sec.png")
plt.close()

# Plot 2: Time comparison
plt.figure(figsize=(10, 6))
plt.plot(benchmark[benchmark["method"] == "naive"]["seq_len"], benchmark[benchmark["method"] == "naive"]["time"], marker='o', label="Naive")
plt.plot(benchmark[benchmark["method"] == "kvcache"]["seq_len"], benchmark[benchmark["method"] == "kvcache"]["time"], marker='s', label="KV Cache")
plt.title("Time vs. Sequence Length")
plt.xlabel("Sequence Length")
plt.ylabel("Time")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("results/benchmark_time.png")
plt.close()

# Plot 3: Speedup ratio
pivot_df = benchmark.pivot(
    index="seq_len", 
    columns="method", 
    values="time"
).reset_index()

pivot_df["speedup"] = pivot_df["naive"] / pivot_df["kvcache"]

plt.figure(figsize=(10, 6))
plt.plot(pivot_df["seq_len"], pivot_df["speedup"], marker='o')
plt.title("Speedup (Naive Time / KV Cache Time) vs. Sequence Length")
plt.xlabel("Sequence Length")
plt.ylabel("Speedup Ratio")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("results/benchmark_speedup.png")
plt.close()

print("All benchmark plots created successfully!")
    