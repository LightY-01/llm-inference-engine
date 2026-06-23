import torch
import time
from model import GPT
from data import Tokenizer
from generate import generate_text

def load_baseline_model(checkpoint_path: str, device: str):
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Retrieve hyperparameters dynamically from the checkpoint
    hyperparams = checkpoint.get('hyperparams', {})
    vocab_size = hyperparams.get('vocab_size', 50257)
    model_dim = hyperparams.get('embed_dim', 512)
    num_heads = hyperparams.get('num_heads', 12)
    num_kv_heads = hyperparams.get('num_kv_heads', 6)
    context_length = hyperparams.get('context_length', 256)
    num_blocks = hyperparams.get('num_blocks', 4)
    
    print(f"Model architecture: vocab_size={vocab_size}, model_dim={model_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, context_length={context_length}, num_blocks={num_blocks}")
    
    # Initialize the model with the exact dimensions from the checkpoint
    model = GPT(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        context_length=context_length,
        num_blocks=num_blocks
    )
    
    # Load the weights into the model
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    
    return model, context_length


device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'gpt_checkpoint.pth' 

# Load Model and context_length
model, context_length = load_baseline_model(checkpoint_path, device)

# Setup Tokenizer
tokenizer = Tokenizer("gpt2") 

prompt = "To be, or not to be"
print(f"\nPrompt: {prompt}")

# Time the Naive Generation
start_time = time.time()

# Pass the prompt to your existing generation loop
output_text = generate_text(
    model=model, 
    prompt=prompt, 
    tokenizer=tokenizer, 
    max_new_tokens=50, 
    context_length=context_length, 
    device=device
)

end_time = time.time()

print(f"\nGenerated:\n{output_text}")
print(f"\nBaseline Generation Time: {end_time - start_time:.4f} seconds")
