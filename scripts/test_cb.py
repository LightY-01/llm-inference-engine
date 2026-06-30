import torch
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

prompt_A = [12, 45, 99, 102, 33]
prompt_B = [12, 45, 99, 102, 33, 55, 66, 77, 88, 99, 11, 22, 33, 44, 55]
num_tokens_to_generate = 5

# Batched Run using InferenceServer
server_batched = InferenceServer(model, device)
req_A_batched = server_batched.add_request(request_id=1, prompt_tokens=prompt_A.copy(), max_new_tokens=num_tokens_to_generate)
req_B_batched = server_batched.add_request(request_id=2, prompt_tokens=prompt_B.copy(), max_new_tokens=num_tokens_to_generate)

# Run the batch until both requests are finished
while len(server_batched.request_queue) > 0:
    server_batched.step()

batched_out_A = req_A_batched.generated_tokens
batched_out_B = req_B_batched.generated_tokens

print(f"Batched Run (5 steps) - A generated: {batched_out_A}")
print(f"Batched Run (5 steps) - B generated: {batched_out_B}")

# Request A alone using InferenceServer
server_A = InferenceServer(model, device)
req_A_solo = server_A.add_request(request_id=1, prompt_tokens=prompt_A.copy(), max_new_tokens=num_tokens_to_generate)
while len(server_A.request_queue) > 0:
    server_A.step()
solo_out_A = req_A_solo.generated_tokens

# Request B alone using InferenceServer
server_B = InferenceServer(model, device)
req_B_solo = server_B.add_request(request_id=2, prompt_tokens=prompt_B.copy(), max_new_tokens=num_tokens_to_generate)
while len(server_B.request_queue) > 0:
    server_B.step()
solo_out_B = req_B_solo.generated_tokens

print(f"Solo Run (5 steps) - A generated: {solo_out_A}")
print(f"Solo Run (5 steps) - B generated: {solo_out_B}")

# Assertions
assert batched_out_A == solo_out_A, f"Mismatch for Request A! Batched: {batched_out_A}, Solo: {solo_out_A}"
assert batched_out_B == solo_out_B, f"Mismatch for Request B! Batched: {batched_out_B}, Solo: {solo_out_B}"

print("success")