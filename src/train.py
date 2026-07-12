import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

class Trainer:
    def __init__(self, model, data_processor, model_name="gpt"):
        self.model = model
        self.data_processor = data_processor
        self.model_name = model_name
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        self.best_val_loss = float('inf')
    
    def get_lr(self, it):
        if it < Config.warmup_iters:
            return Config.learning_rate * it / Config.warmup_iters
        if it > Config.lr_decay_iters:
            return Config.min_lr
        decay_ratio = (it - Config.warmup_iters) / (Config.lr_decay_iters - Config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return Config.min_lr + coeff * (Config.learning_rate - Config.min_lr)
    
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
            lr = self.get_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
                losses = self.estimate_loss()
                train_ppl = math.exp(losses['train'])
                val_ppl = math.exp(losses['val'])
                
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    
                params = sum(p.numel() for p in self.model.parameters())
                print(f"[{self.model_name}] step {iter:5d}/{Config.max_iters} | "
                      f"train loss {losses['train']:.4f} (ppl {train_ppl:.2f}) | "
                      f"val loss {losses['val']:.4f} (ppl {val_ppl:.2f}) | "
                      f"lr {lr:.2e} | params {params/1e6:.1f}M")
                
                save_path = Config.model_save_path.format(self.model_name, iter)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': Config.__dict__,
                    'iter': iter,
                    'val_loss': losses['val'],
                    'train_loss': losses['train'],
                }, save_path)
            
            xb, yb = self.data_processor.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if Config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            
            self.optimizer.step()
    
    def train_bigram(self, bigram_model):
        bigram_optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
        best_loss = float('inf')
        
        print(f"\nTraining Bigram baseline...")
        for iter in range(2000):
            xb, yb = self.data_processor.get_batch('train')
            logits, loss = bigram_model(xb, yb)
            bigram_optimizer.zero_grad()
            loss.backward()
            bigram_optimizer.step()
            
            if iter % 500 == 0 or iter == 1999:
                Xv, Yv = self.data_processor.get_batch('val')
                _, val_loss = bigram_model(Xv, Yv)
                ppl = math.exp(val_loss.item())
                print(f"  [bigram] step {iter:5d}/2000 | val loss {val_loss.item():.4f} (ppl {ppl:.2f})")
                if val_loss < best_loss:
                    best_loss = val_loss
        
        params = sum(p.numel() for p in bigram_model.parameters())
        save_path = Config.bigram_save_path
        torch.save({
            'model_state_dict': bigram_model.state_dict(),
            'val_loss': best_loss,
        }, save_path)
        
        return best_loss
