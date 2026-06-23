import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Dict, List, Tuple
import tiktoken

def create_batches(data: TensorType[int], batch_size: int, context_length: int) -> Tuple[TensorType[int], TensorType[int]]:
    """Samples random chunks from the dataset to form a batch."""
    # Generate random starting indices for each batch
    ix = torch.randint(0, len(data) - context_length, (batch_size,))
    # Create input (x) and target (y) tensors by slicing the data
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+1+context_length] for i in ix])
    return x, y

class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)
