import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def download_and_prepare():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if os.path.exists(Config.input_path):
        print(f"Processed file already exists: {Config.input_path}")
        return

    print(f"Downloading Havadis dataset (up to {Config.havadis_max_articles:} articles)...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    ds = load_dataset("turkish-nlp-suite/Havadis", split="train", streaming=True)

    count = 0
    with open(Config.havadis_raw_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= Config.havadis_max_articles:
                break

            raw_text = example.get("text", "").strip()
            if not raw_text or len(raw_text) < 50:
                continue

            lines = raw_text.split("\n", 1)
            title = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else ""

            if not title or not text or len(text) < 50:
                continue

            title_clean = title.replace("\n", " ").replace("\r", " ")
            text_clean = text.replace("\r", "")

            f.write(f"Başlık: {title_clean}\n")
            f.write(f"İçerik: {text_clean}\n")
            f.write("HABER SONU\n\n")
            count += 1

            if count % 10000 == 0:
                print(f"  {count} articles processed...")

    print(f"\nDownloaded {count} articles to {Config.havadis_raw_path}")
    print(f"File size: {os.path.getsize(Config.havadis_raw_path) / 1e6:.1f} MB")

    print(f"\nCopying to {Config.input_path}...")
    with open(Config.havadis_raw_path, "r", encoding="utf-8") as src:
        with open(Config.input_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    print("Done! Ready for training.")

if __name__ == "__main__":
    download_and_prepare()
