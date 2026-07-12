#!/bin/bash
set -e

echo "GPT-from-Scratch + Havadis Turkish News"
echo ""

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install -e . -q
pip install datasets -q

# 2. Download and prepare Havadis dataset
echo ""
echo "[2/4] Downloading Havadis dataset..."
python scripts/download_and_prepare.py

# 3. Train model (bigram baseline + GPT)
echo ""
echo "[3/4] Training models..."
python src/main.py

# 4. Done
echo ""
echo "[4/4] Complete!"
echo "Results saved to outputs/"
echo ""
echo "Generated samples:"
ls -la outputs/
cat outputs/results.txt
