#!/bin/bash
set -e

echo "=== BayarWoy NLP Service Setup ==="

# Install python3-pip dan python3-venv jika belum ada
if ! python3 -m pip --version &>/dev/null; then
    echo "Menginstall python3-pip..."
    sudo apt-get install -y python3-pip python3-venv
fi

# Buat virtual environment
if [ ! -d ".venv" ]; then
    echo "Membuat virtual environment..."
    python3 -m venv .venv
fi

# Aktifkan dan install dependencies
echo "Menginstall dependencies..."
.venv/bin/pip install -r requirements.txt -q

# Train model
echo "Melatih model klasifikasi..."
.venv/bin/python src/train.py

echo ""
echo "Setup selesai! Jalankan service dengan:"
echo "  .venv/bin/uvicorn src.main:app --reload --port 8001"