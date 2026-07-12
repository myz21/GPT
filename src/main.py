import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.data import DataProcessor
from src.model import GPTLanguageModel
from src.bigram import BigramModel
from src.train import Trainer

def generate_samples(model, data_processor, prompt="", max_tokens=300, temperature=0.8, top_k=50):
    model.eval()
    
    if prompt:
        context = torch.tensor([data_processor.encode(prompt)], dtype=torch.long, device=Config.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    
    for _ in range(max_tokens):
        idx_cond = context[:, -Config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
    
    return data_processor.decode(context[0].tolist())

def print_results(model_name, val_loss, params, gen_text):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Parameters: {params/1e6:.2f}M")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Perplexity: {math.exp(val_loss):.2f}")
    print(f"  Generated Text:")
    print(f"  {gen_text[:2000]}")
    print(f"{'='*60}\n")

def main():
    torch.manual_seed(Config.seed)
    
    print(f"Device: {Config.device}")
    print(f"Mode: {Config.mode} ({Config.MODEL_CONFIGS[Config.mode]['desc']})")
    print(f"Dataset: {Config.dataset}")
    print()
    
    data_processor = DataProcessor(Config.input_path, val_split=Config.havadis_val_split)
    
    print(f"\n{'-'*60}")
    print("BASELINE 1: Random (uniform prediction)")
    print(f"{'-'*60}")
    random_loss = math.log(data_processor.vocab_size)
    print(f"  Random loss: {random_loss:.4f} (ppl {math.exp(random_loss):.2f})")
    
    print(f"\n{'-'*60}")
    print("BASELINE 2: Bigram Model")
    print(f"{'-'*60}")
    bigram = BigramModel(data_processor.vocab_size).to(Config.device)
    bigram_params = sum(p.numel() for p in bigram.parameters())
    trainer = Trainer(bigram, data_processor, model_name="bigram")
    bigram_val_loss = trainer.train_bigram(bigram)
    
    bigram_gen = generate_samples(bigram, data_processor, prompt="Başlık: ", max_tokens=200)
    print_results("Bigram Baseline", bigram_val_loss, bigram_params, bigram_gen)
    
    print(f"\n{'-'*60}")
    print(f"GPT Model ({Config.mode})")
    print(f"{'-'*60}")
    model = GPTLanguageModel(data_processor.vocab_size).to(Config.device)
    gpt_params = sum(p.numel() for p in model.parameters())
    print(f"  {gpt_params/1e6:.1f}M parameters")
    
    trainer = Trainer(model, data_processor, model_name=f"gpt_{Config.mode}")
    trainer.train()
    
    model.eval()
    with torch.no_grad():
        Xv, Yv = data_processor.get_batch('val')
        _, gpt_loss = model(Xv, Yv)
    
    gpt_prompts = [
        "Başlık: Yapay Zeka Türkiye'de",
        "Başlık: Ekonomi",
        "Başlık: Spor",
        "",
    ]
    gpt_gen = ""
    for prompt in gpt_prompts:
        gpt_gen += f"\n--- Prompt: '{prompt if prompt else '(zero context)'}' ---\n"
        gen = generate_samples(model, data_processor, prompt=prompt, max_tokens=300)
        gpt_gen += gen[:600] + "\n"
    
    print_results(f"GPT-{Config.mode.capitalize()}", gpt_loss.item(), gpt_params, gpt_gen)
    
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Loss':<10} {'PPL':<10} {'Params':<10}")
    print(f"  {'-'*50}")
    print(f"  {'Random':<20} {random_loss:<10.4f} {math.exp(random_loss):<10.2f} {'0':<10}")
    print(f"  {'Bigram':<20} {bigram_val_loss:<10.4f} {math.exp(bigram_val_loss):<10.2f} {bigram_params/1e6:<10.2f}M")
    print(f"  {'GPT-'+Config.mode.capitalize():<20} {gpt_loss.item():<10.4f} {math.exp(gpt_loss.item()):<10.2f} {gpt_params/1e6:<10.2f}M")
    print(f"{'='=60}")
    
    comparison = f"""
FINAL RESULTS:
=============
Dataset: {Config.dataset} ({Config.havadis_max_articles} articles)
Mode:    {Config.mode}
Device:  {Config.device}

{'Model':<20} {'Val Loss':<10} {'Perplexity':<12} {'Params':<10}
{'-'*52}
{'Random':<20} {random_loss:<10.4f} {math.exp(random_loss):<12.2f} {'0':<10}
{'Bigram':<20} {bigram_val_loss:<10.4f} {math.exp(bigram_val_loss):<12.2f} {bigram_params/1e6:<10.2f}M
{'GPT-'+Config.mode.capitalize():<20} {gpt_loss.item():<10.4f} {math.exp(gpt_loss.item()):<12.2f} {gpt_params/1e6:<10.2f}M
"""
    with open(Config.results_path, "w") as f:
        f.write(comparison)
    print(f"\nResults saved to {Config.results_path}")

if __name__ == "__main__":
    main()
