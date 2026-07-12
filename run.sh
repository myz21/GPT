#!/bin/bash

shutdown_pod() {
    echo ""
    echo "=== RUN FINISHED (exit code: $?) ==="
    echo "Shutting down RunPod in 10 seconds..."
    sleep 10
    sudo shutdown -h now
}

trap shutdown_pod EXIT

set -e

cd /root/GPT

echo "=== Step 1: Install dependencies ==="
pip install -e . -q
pip install datasets -q

echo "=== Step 2: Download dataset (744K articles) ==="
python scripts/download_and_prepare.py --max-articles 744000

echo "=== Step 3: Validate dataset ==="
pytest tests/test_dataset.py -v

echo "=== Step 4: Train models ==="
python src/main.py

echo "=== Step 5: Run all tests ==="
pytest tests/ -v
