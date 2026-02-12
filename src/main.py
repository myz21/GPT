import torch
import os
import sys

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.data import DataProcessor
from src.model import GPTLanguageModel
from src.train import Trainer

def main():
    torch.manual_seed(Config.seed)
    
    # Load data
    data_processor = DataProcessor(Config.input_path)
    
    # Create model
    model = GPTLanguageModel(data_processor.vocab_size).to(Config.device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Train
    trainer = Trainer(model, data_processor)
    trainer.train()
    
    # Generate
    context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    generated = model.generate(context, max_new_tokens=2000)
    print(data_processor.decode(generated[0].tolist()))
    
    # Save
    filename = os.path.basename(Config.input_path).split('.')[0]
    save_path = Config.model_save_path.format(filename)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    main()
