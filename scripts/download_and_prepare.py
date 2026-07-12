import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def download_and_prepare(small=False, max_articles=None):
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if max_articles is None:
        max_articles = Config.havadis_small_max_articles if small else Config.havadis_max_articles
    raw_path = Config.havadis_small_path if small else Config.havadis_raw_path
    label = "small (quick test)" if small else f"full ({max_articles:,} articles)"

    if os.path.exists(raw_path):
        print(f"{label} dataset already exists: {raw_path}")
        return

    print(f"Downloading Havadis dataset ({label}, up to {max_articles:,} articles)...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    ds = load_dataset("turkish-nlp-suite/Havadis", split="train", streaming=True)

    count = 0
    with open(raw_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_articles:
                break

            url = example.get("url", "")
            raw_text = example.get("text", "").strip()

            if not raw_text or len(raw_text) < 50 or not url:
                continue

            slug = url.rstrip("/").split("/")[-1]
            title = slug.replace("-", " ")

            text_clean = raw_text.replace("\r", "")

            f.write(f"Başlık: {title}\n")
            f.write(f"İçerik: {text_clean}\n")
            f.write("HABER SONU\n\n")
            count += 1

            if count % 10000 == 0:
                print(f"  {count} articles processed...")

    print(f"\nDownloaded {count} articles to {raw_path}")
    print(f"File size: {os.path.getsize(raw_path) / 1e6:.1f} MB")

    if not small:
        print(f"\nCopying to {Config.input_path}...")
        with open(raw_path, "r", encoding="utf-8") as src:
            with open(Config.input_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

    print("Done! Ready for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Download 10K articles for quick testing")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to download (overrides default)")
    args = parser.parse_args()
    download_and_prepare(small=args.small, max_articles=args.max_articles)
