# GPT-from-Scratch: Learning Transformers with Turkish Text

A character-level Decoder-only Transformer (GPT architecture) built from scratch in PyTorch. Follows Andrej Karpathy's nanoGPT approach, adapted for Turkish language modeling.

## Project Structure

```text
GPT/
├── config/
│   └── config.py          # Hyperparameters (4 modes: tiny/small/medium/large)
├── data/
│   ├── data.txt           # Test dataset
│   └── havaids_*.txt      # Havadis Turkish news corpus (automatically downloaded)
├── scripts/
│   ├── download_and_prepare.py  # Download Havadis from HuggingFace
│   └── run_all.sh               # Single command for full pipeline
├── notebooks/
│   ├── GPT_DEV.ipynb      # Development notebook
│   ├── demo.ipynb         # Demo: attention viz + generation samples
│   └── initial.ipynb
├── outputs/
│   └── model_*.pth        # Model checkpoints
├── src/
│   ├── main.py            # Entry point: trains bigram + GPT, compares results
│   ├── model.py           # GPT architecture (Head, MHA, Block, GPTLanguageModel)
│   ├── bigram.py          # Bigram baseline model for comparison
│   ├── data.py            # DataProcessor class
│   └── train.py           # Trainer with LR scheduler, gradient clipping, perplexity
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
git clone https://github.com/myz21/GPT.git
cd GPT
pip install -e .
pip install datasets   # for Havadis download

# Full pipeline: download Havadis → train bigram + GPT → compare
bash scripts/run_all.sh

# Or step by step:
python scripts/download_and_prepare.py
python src/main.py
```

## Model Modes

| Mode | Parameters | VRAM | Training Time (RTX A6000) |
|------|-----------|------|--------------------------|
| tiny | ~107K | ~0.5 GB | ~5 min |
| small | ~1.2M | ~1 GB | ~20 min |
| medium | ~4.8M | ~2 GB | ~1.5 hr |
| large | ~16M | ~4 GB | ~3 hr |

Set mode in `config/config.py`: `mode = "medium"`

## Benchmark Results

```
| Model  | Val Loss | Perplexity | Parameters |
|--------|----------|------------|------------|
| Random | ~4.5     | ~90        | 0          |
| Bigram | ~3.0     | ~20        | ~65K       |
| GPT    | ~1.5     | ~4.5       | ~4.8M      |
```

## Dataset: Havadis Turkish News Corpus

- **Source:** [Havadis](https://huggingface.co/datasets/turkish-nlp-suite/Havadis) on HuggingFace
- **Size:** 744K news articles, 2.88 GB, 315M words (uses 100K by default for fast training)
- **Sources:** CNN Türk, Hürriyet, Milliyet, Sözcü, Sabah, NTV, Star, Posta
- **Format:** `Başlık: <title>\nİçerik: <body>\nHABER SONU\n`

RunPod (RTX A6000) training: ~1.5 hours → val loss ~1.5

## Features

- Causal multi-head self-attention with masked attention
- Residual connections + layer normalization
- Cosine learning rate schedule with linear warmup
- Gradient clipping for stable training
- Perplexity monitoring (train + validation)
- Checkpoint saving at each evaluation step
- Top-k / temperature sampling for generation
- Bigram baseline for comparison
- Attention pattern visualization

## About

Built to understand transformers from the inside out. Every layer is written manually — no `nn.Transformer`, no black boxes. The project demonstrates how attention mechanisms, feed-forward networks, and autoregressive generation work at the character level.

Inspired by Andrej Karpathy's [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY).

License: MIT
