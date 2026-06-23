import torch
import torch.nn.functional as F
from data import Tokenizer, create_batches
from model import GPT
import math
import matplotlib.pyplot as plt
from generate import generate_text

# --- parameters ---
# 63,149,137 parameters
# vocab_size = 50257
batch_size     = 64
context_length = 256
embed_dim      = 512
num_heads      = 12
num_kv_heads   = 6
learning_rate  = 3e-4
warmup_steps   = 100
max_iters      = 5000
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters     = 50
CHECKPOINT     = 'gpt_checkpoint.pth'

def get_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * 0.1 + learning_rate * 0.9 * coeff

def train_step(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, max_norm: float = 1.0) -> float:
    """A single training step, correctly handling both CPU and CUDA."""
    device_type = X.device.type
    use_amp = device_type == 'cuda'

    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    
    return loss.item()

@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, context_length: int, device: str) -> Dict[str, float]:
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = create_batches(data, batch_size, context_length)
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            vocab_size = logits.shape[-1]
            loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def plot_curves(steps, train_perplexities, val_perplexities):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_perplexities, label='Train Perplexity', color='blue')
    plt.plot(steps, val_perplexities, label='Validation Perplexity', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity (Lower is better)')
    plt.title('GPT Training & Validation Perplexity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('perplexity_curves.png')
    plt.close()


with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
vocab = {ch: i for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size}")
print(f"Training on device: {device}")

tokenizer = Tokenizer("gpt2")
full_data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data   = full_data[n:]

model     = GPT(vocab_size, embed_dim, num_heads, num_kv_heads, context_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scaler    = torch.amp.GradScaler() if device == 'cuda' else None

tracked_steps      = []
train_ppl_history  = []
val_ppl_history    = []

for iter in range(max_iters):
    lr = get_lr(iter, max_iters, learning_rate, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x_batch, y_batch = create_batches(train_data, batch_size, context_length)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    loss = train_step(model, x_batch, y_batch, optimizer, scaler)

    if iter % 500 == 0 or iter == max_iters - 1:
        losses    = estimate_loss(model, train_data, val_data, batch_size, context_length, device)
        train_ppl = math.exp(losses['train'])
        val_ppl   = math.exp(losses['val'])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"         train ppl  {train_ppl:.4f}, val ppl  {val_ppl:.4f}")
        tracked_steps.append(iter)
        train_ppl_history.append(train_ppl)
        val_ppl_history.append(val_ppl)
        plot_curves(tracked_steps, train_ppl_history, val_ppl_history)
        # CHECKPOINT SAVING
        # Save everything infer.py needs: weights, vocab, and the exact
        # hyperparameters used, so the architecture can be reconstructed exactly
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'hyperparams': {
                'vocab_size':     vocab_size,
                'embed_dim':      embed_dim,
                'num_heads':      num_heads,
                'num_kv_heads':   num_kv_heads,
                'context_length': context_length,
                'num_blocks':     4,
            }
        }, CHECKPOINT)
        print(f"\nCheckpoint saved → {CHECKPOINT}")
print("\nTraining Complete.")
