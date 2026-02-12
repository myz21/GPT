# GPT-from-Scratch: Learning Transformers with Turkish Text

A character-level Decoder-only Transformer (GPT architecture) from scratch. This project follows the spirit of Andrej Karpathy's nanoGPT tutorial, adapted for Turkish language modeling.

## Project Structure

```text
GPT/
├── config/
│   └── config.py          # Hyperparameters and configuration
├── data/
│   └── data.txt           # Training data
├── notebooks/
│   ├── GPT_DEV.ipynb      # Development notebook
│   └── initial.ipynb      # Initial exploration
├── outputs/
│   └── test_model.pth     # Saved model checkpoints
├── src/
│   ├── __init__.py
│   ├── main.py            # Main entry point
│   ├── model.py           # GPT transformer architecture
│   ├── data.py            # DataProcessor class
│   └── train.py           # Trainer class
├── pyproject.toml         # Project configuration
└── README.md
```

> **Status:** Modular architecture implemented! Code is organized into reusable modules with clear separation of concerns.

## 🛠️ Setup & Usage

### Quick Start

```bash
git clone https://github.com/myz21/GPT.git
cd GPT

# Option 1: Using pip (standard)
pip install -e .

# Option 2: Using uv (faster)
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync
```

### Run Training

```bash
python src/main.py
```

### Module Overview

| Module | Purpose |
|--------|---------|
| `config/config.py` | Hyperparameters (batch_size, block_size, learning_rate, etc.) |
| `src/data.py` | `DataProcessor` - loads text, tokenizes, creates batches |
| `src/model.py` | GPT architecture - `Head`, `MultiHeadAttention`, `Block`, `GPTLanguageModel` |
| `src/train.py` | `Trainer` class - handles training loop and loss estimation |
| `src/main.py` | Main entry point - orchestrates data loading, model training, generation |

### Google Colab (Original Method)

1. Upload `nutuk.txt` to your Google Drive
2. Open `notebooks/GPT_DEV.ipynb` in Colab
3. Adjust file paths and run cells

## 📊 Model Specifications

**Production Configuration:**
* **Context Window:** 256 characters
* **Embedding Dimension:** 256
* **Attention Heads:** 6
* **Transformer Blocks:** 6
* **Batch Size:** 64
* **Parameters:** ~4.8M
* **Dropout:** 0.2

**Test/Development Configuration:**
* **Context Window:** 32 characters
* **Embedding Dimension:** 64
* **Attention Heads:** 2
* **Transformer Blocks:** 2
* **Batch Size:** 4
* **Parameters:** ~107K
* **Device:** CPU (for testing without GPU)

## 🚀 Roadmap

### ✅ Completed
- [x] Basic transformer architecture (self-attention, FFN, residual connections)
- [x] Character-level tokenization for Turkish
- [x] Training loop with AdamW optimizer
- [x] Text generation with temperature sampling
- [x] **Modularize codebase** ✨
  - [x] `src/model.py` - Transformer architecture
  - [x] `src/data.py` - Dataset and batching
  - [x] `src/train.py` - Training class
  - [x] `src/main.py` - Main entry point
  - [x] `config/config.py` - Centralized configuration
  - [x] `pyproject.toml` - Project metadata

### 🔨 In Progress
- [ ] Hyperparameter tuning for better convergence
- [ ] Add validation metrics and monitoring
- [ ] Implement learning rate scheduler

### 🎓 Learning Goals
- [ ] Implement learning rate scheduler (cosine decay with warmup)
- [ ] Add gradient clipping and norm monitoring
- [ ] Visualize attention patterns
- [ ] Experiment with different positional encodings
- [ ] Try weight tying (embedding ↔ output projection)

### 🔬 Advanced Experiments (Future)
- [ ] Compare character-level vs. BPE tokenization
- [ ] Test Flash Attention for efficiency
- [ ] Implement KV caching for faster inference
- [ ] Scale up to larger Turkish corpora

## 📈 Training Tips

**Hardware:**
* Works on free Colab GPUs (T4)
* Training time: ~30-60 minutes for decent results
* Can be trained on CPU (much slower)

**Hyperparameters to experiment with:**
* `batch_size`: 32-64 (depends on GPU memory)
* `learning_rate`: 1e-3 to 3e-4
* `block_size`: 128-512 (longer = better context, more memory)
* `n_layer`: 4-8 (deeper = more capacity, slower training)

## 🤝 Acknowledgments

This project is heavily inspired by:
* [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
* [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial

## 📝 License

MIT License - feel free to use for learning and experimentation!

---

**Note:** This is a work-in-progress learning project. Contributions and suggestions are welcome, especially from those also learning transformers! 🚀
