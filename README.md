# Custom LLM Inference Engine

A PyTorch-based Transformer inference engine built entirely from scratch. The goal of this project is to optimize a custom GPT codebase by implementing the core serving mechanisms used by production engines like vLLM, TGI, and Ollama. 

Inference optimization is a critical engineering challenge where memory bandwidth, rather than pure compute, is the primary bottleneck. This project tackles that bottleneck through three major architectural upgrades.

## Project Roadmap

- **Phase 1: KV-Caching (Complete)**
Pre-allocating key/value tensor buffers to avoid recomputing past states during autoregressive generation.
  
- **Phase 2: Continuous Batching**
Replacing static batching with a dynamic request manager that queues and batches concurrent requests at every single decode step, utilizing padding masks to handle variable-length sequences.

- **Phase 3: Speculative Decoding**
Deploying a secondary, scaled-down "draft" model to auto-regressively propose $K$ future tokens, which the primary model verifies in a single forward pass using rejection sampling. 

## Benchmarking
The ultimate efficiency of this engine will be measured against standard sequential serving and benchmarked against HuggingFace's native `generate()` function.

## Quickstart
*(Instructions for loading the model and running the inference server will be added here as the engine is built.)*
