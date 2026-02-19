# DNA Language Model

## Objectives

- Create a custom transformer using Pytorch.
- Pre-train a BERT-style transformer on the Human Reference Genome using masked language modelling and span masking. 
- Apply mixed-precision training, FlashAttention, and gradient checkpointing for efficient training; benchmarked post-training quantisation, structured pruning and knowledge distillation for inference.
- Compared against alternative transformer architectures on Genomic Benchmarks classification tasks with Hugging Face models.
- Implement custom CUDA kernels for fused tokenisation and RoPE.
- Track experiments tracked with W&B.

