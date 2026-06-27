import pandas as pd
import matplotlib.pyplot as plt

benchmark = pd.read_csv("results/benchmark_results.csv")

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