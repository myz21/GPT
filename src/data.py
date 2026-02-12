import torch
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

class DataProcessor:
    def __init__(self, text_path):
        with open(text_path) as f:
            self.text = f.read()
        
        # Character encoding
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # Encode/Decode functions
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])
        
        # Train-val split
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
        x = torch.stack([data[i:i+Config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
        x, y = x.to(Config.device), y.to(Config.device)
        return x, y
