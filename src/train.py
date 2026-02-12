import torch
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

class Trainer:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(Config.eval_iters)
            for k in range(Config.eval_iters):
                X, Y = self.data_processor.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        for iter in range(Config.max_iters):
            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.data_processor.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()