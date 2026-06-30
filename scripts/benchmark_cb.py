import torch
import time
import random
import matplotlib.pyplot as plt
from model import GPT
from inference_server import InferenceServer


torch.manual_seed(42)

model = GPT(
    vocab_size=50257,
    model_dim=512,
    num_heads=12,
    num_kv_heads=6,
    context_length=256,
    num_blocks=4
)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

batch_sizes = [1, 2, 4, 8]
total_requests = 100

# 100 requests with different prompt lengths (10 to 200)
# Adjust random lengths to ensure total length <= context_length (256)
prompt_lengths = [random.randint(10, 128) for _ in range(total_requests)]
# randomized max_new_tokens (20 to 128)
max_new_tokens = [random.randint(20, 128) for _ in range(total_requests)]
total_tokens = sum(max_new_tokens)

prompts = [[random.randint(0, 50256) for _ in range(prompt_lengths[i])] for i in range(total_requests)]

results = {}

for batch_size in batch_sizes:
    server = InferenceServer(model, device, max_batch_size=batch_size)

    for i in range(total_requests):
        server.add_request(request_id=i, prompt_tokens=prompts[i], max_new_tokens=max_new_tokens[i])
    
    start = time.time()
    while len(server.request_queue) > 0:
        server.step()
    end = time.time()
    print(f"Batch size {batch_size}: {end - start}")

    metric = total_tokens / (end - start)
    results[batch_size] = metric

plt.bar(results.keys(), results.values())
plt.xlabel("Batch Size")
plt.ylabel("Throughput (tokens/second)")
plt.title("Inference Benchmark - Comparison of Batch Sizes")
plt.savefig('results/cont_batch_benchmark.png')
plt.close()




    
    
    

    