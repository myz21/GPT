import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


class DataProcessor:
    def __init__(self, text_path, val_split=0.1):
        with open(text_path, encoding="utf-8") as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda lst: ''.join([self.itos[i] for i in lst])

        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int((1 - val_split) * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        print(f"Data: {len(data):,} chars, vocab: {self.vocab_size}")
        print(f"Train: {len(self.train_data):,}, Val: {len(self.val_data):,}")

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
        x = torch.stack([data[i:i+Config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
        x, y = x.to(Config.device), y.to(Config.device)
        return x, y
