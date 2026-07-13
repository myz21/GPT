import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


import re


_CONTROL_CHARS = set(chr(i) for i in range(32) if chr(i) not in '\n\t\r')
_EMOJI_PATTERN = re.compile(
    '[\U0001F000-\U0001FFFF]'
    '|\U0000FE00|\U0000FE0F'
    '|\U0000200D|\U0000200C'
    '|[\U000025A0-\U000027BF]'
    '|[\U00002900-\U00002BFF]'
    '|[\U00002300-\U000024FF]'
    '|[\U00002100-\U0000214F]'
    '|[\U00002C60-\U00002C7F]'
    '|[\U0001F600-\U0001F64F]'
    '|[\U0001F300-\U0001F5FF]'
    '|[\U0001F680-\U0001F6FF]'
    '|[\U0001F900-\U0001F9FF]'
    '|[\U000020A0-\U000020CF]'
    '|[\U0000E000-\U0000F8FF]'
    '|[\U000F0000-\U000FFFFD]'
    '|\U000100000-\U0010FFFD',
)


def clean_text(text):
    text = ''.join(ch for ch in text if ch not in _CONTROL_CHARS)
    text = _EMOJI_PATTERN.sub('', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return text


class DataProcessor:
    def __init__(self, text_path, val_split=0.1):
        with open(text_path, encoding="utf-8") as f:
            raw_text = f.read()

        self.text = clean_text(raw_text)
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
