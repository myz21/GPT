# Character-Level GPT Model

A minimal implementation of GPT (Generative Pre-trained Transformer) for character-level language modeling, built from scratch using PyTorch.

## Overview

This project implements a character-level language model based on the GPT (Generative Pre-trained Transformer) architecture. Unlike word-level or subword tokenization, this model operates directly on individual characters, making it simpler and more flexible for various text generation tasks.

## Features

- **Pure PyTorch Implementation**: Built from scratch with no external dependencies except PyTorch and NumPy
- **Character-Level Tokenization**: Simple tokenization that works with any text
- **Multi-Head Self-Attention**: Implements the core transformer attention mechanism
- **Positional Embeddings**: Encodes position information for sequence modeling
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enables training of deeper networks
- **Text Generation**: Supports temperature-based sampling and top-k sampling

## Architecture

The model consists of:
- Token embedding layer
- Positional embedding layer
- Multiple transformer blocks, each containing:
  - Multi-head self-attention mechanism
  - Feed-forward neural network
  - Layer normalization
  - Residual connections
- Output projection layer

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+

## Usage

### Basic Training

```python
from train import train, generate_text
import torch

# Prepare your text
text = """Your training text goes here..."""

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train the model
model, tokenizer = train(
    text,
    n_embd=384,        # Embedding dimension
    num_heads=6,       # Number of attention heads
    num_layers=6,      # Number of transformer layers
    block_size=256,    # Maximum context length
    batch_size=64,     # Batch size
    max_iters=5000,    # Training iterations
    learning_rate=3e-4,
    device=device
)

# Generate text
generated = generate_text(
    model, tokenizer, 
    prompt="Your prompt", 
    max_new_tokens=500,
    temperature=0.8,
    device=device
)
print(generated)
```

### Running the Training Script

```bash
python train.py
```

This will train a model on sample text and generate some examples.

### Running Examples

```bash
python example.py
```

This demonstrates different use cases including Shakespeare-style text generation.

## Model Components

### 1. `model.py` - Core Model Architecture

- **Head**: Single attention head
- **MultiHeadAttention**: Multiple attention heads running in parallel
- **FeedForward**: Position-wise feed-forward network
- **Block**: Complete transformer block
- **GPTLanguageModel**: Main model class

### 2. `train.py` - Training and Utilities

- **CharacterTokenizer**: Simple character-level tokenizer
- **train()**: Main training function
- **generate_text()**: Text generation function
- **get_batch()**: Mini-batch generation
- **estimate_loss()**: Loss estimation on train/val sets

### 3. `example.py` - Usage Examples

Demonstrates different use cases and configurations.

## Hyperparameters

Key hyperparameters you can tune:

- `n_embd`: Embedding dimension (default: 384)
- `num_heads`: Number of attention heads (default: 6)
- `num_layers`: Number of transformer blocks (default: 6)
- `block_size`: Maximum context length (default: 256)
- `batch_size`: Training batch size (default: 64)
- `learning_rate`: Learning rate (default: 3e-4)
- `dropout`: Dropout rate (default: 0.2)

## Text Generation

The model supports several generation strategies:

```python
# Basic generation
generated = model.generate(context, max_new_tokens=100)

# With temperature (higher = more random)
generated = model.generate(context, max_new_tokens=100, temperature=0.8)

# With top-k sampling (only sample from k most likely tokens)
generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=10)
```

## Training Tips

1. **Start Small**: Begin with smaller models (fewer layers, smaller embedding dimension) to iterate quickly
2. **Monitor Loss**: Watch both training and validation loss to detect overfitting
3. **Adjust Learning Rate**: If loss plateaus, try reducing the learning rate
4. **Increase Data**: More training data generally leads to better results
5. **Context Length**: Longer `block_size` allows the model to capture longer dependencies but requires more memory

## Example Output

After training on Shakespeare text:

```
Prompt: 'First'
Generated: First Citizen:
We know't, we know the people.

All:
Resolved. resolved.

First Citizen:
Let us kill him...
```

## Project Structure

```
GPT/
├── model.py          # GPT model architecture
├── train.py          # Training script and utilities
├── example.py        # Usage examples
├── requirements.txt  # Python dependencies
├── .gitignore       # Git ignore file
└── README.md        # This file
```

## How It Works

1. **Tokenization**: Text is converted to a sequence of integer indices (one per character)
2. **Embedding**: Each character index is converted to a dense vector
3. **Positional Encoding**: Position information is added to each token
4. **Transformer Blocks**: Self-attention and feed-forward layers process the sequence
5. **Prediction**: Output layer predicts the next character
6. **Generation**: Sample from predictions to generate new text

## License

MIT License - Feel free to use this code for learning and experimentation.

## Acknowledgments

Based on the GPT architecture from the paper "Attention is All You Need" and subsequent GPT models by OpenAI. Inspired by Andrej Karpathy's educational content on transformers and language models.
