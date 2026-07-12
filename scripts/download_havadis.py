import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from config.config import Config


def download_havadis():
    os.makedirs(os.path.dirname(Config.havadis_raw_path), exist_ok=True)

    print("Downloading Havadis dataset from HuggingFace...")
    ds = load_dataset("turkish-nlp-suite/Havadis", split="train", streaming=True)

    total = len(ds) if hasattr(ds, '__len__') else 744868
    print(f"Total instances: {total}")

    count = 0
    with open(Config.havadis_raw_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            title = example.get("title", "").strip()
            text = example.get("text", "").strip()
            source = example.get("source", "").strip()

            if not title or not text:
                continue

            f.write(f"Başlık: {title}\n")
            if source:
                f.write(f"Kaynak: {source}\n")
            f.write(f"İçerik: {text}\n")
            f.write("HABER SONU\n\n")
            count += 1

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{total} articles...")

    print(f"\nDone! {count} articles saved to {Config.havadis_raw_path}")
    return count

if __name__ == "__main__":
    download_havadis()
