# GPT-from-Scratch: Learning Transformers with Turkish Text

A character-level Decoder-only Transformer (GPT architecture) from scratch. This project follows the spirit of Andrej Karpathy's nanoGPT tutorial, adapted for Turkish language modeling.

## Project Structure

```text
GPT/
├── data/                  # Training data (nutuk.txt - not in repo)
├── notebooks/             
│   └── GPT_DEV.ipynb     # Main development notebook (Google Colab)
├── outputs/               # Model checkpoints (generated during training)
├── src/                   # Source code (to be modularized from notebook)
└── README.md
```

> **Current Status:** Most of the implementation is in `GPT_DEV.ipynb`. The `src/` directory is planned for refactoring the notebook into reusable modules.

## 🛠️ Setup & Usage

### Running on Google Colab (Recommended)

1. Upload `nutuk.txt` to your Google Drive
2. Open `notebooks/GPT_DEV.ipynb` in Colab
3. Mount your Drive and adjust the file path:
   ```python
   with open("/content/drive/MyDrive/nutuk.txt") as f:
       text = f.read()
   ```
4. Run all cells to train the model

### Local Setup

```bash
git clone https://github.com/myz21/GPT.git
cd GPT
pip install torch numpy matplotlib tqdm
```

## 📊 Model Specifications

**Current Implementation:**
* **Dataset:** Nutuk by M.K. Atatürk (1,577,732 characters)
* **Tokenization:** Character-level (Turkish alphabet)
* **Context Window:** 256 characters
* **Embedding Dimension:** 256
* **Attention Heads:** 6
* **Transformer Blocks:** 6
* **Parameters:** ~4.8M
* **Dropout:** 0.2

## 🚀 Roadmap

### ✅ Completed
- [x] Basic transformer architecture (self-attention, FFN, residual connections)
- [x] Character-level tokenization for Turkish
- [x] Training loop with AdamW optimizer
- [x] Text generation with temperature sampling

### 🔨 In Progress
- [ ] **Modularize codebase:** Move notebook code to `src/` directory
  - `src/model.py` - Transformer architecture
  - `src/data_loader.py` - Dataset and batching
  - `src/train.py` - Training script
  - `src/generate.py` - Inference utilities

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
