#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

echo "=========================================="
echo "      Setting up RL-IDS Environment       "
echo "=========================================="

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment in ./$VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "[*] Virtual environment already exists."
fi

# Activate venv
source $VENV_DIR/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "Python_code/requirements.txt" ]; then
    echo "[*] Installing dependencies from Python_code/requirements.txt..."
    pip install -r Python_code/requirements.txt
else
    echo "[!] Error: Python_code/requirements.txt not found!"
    exit 1
fi

echo "=========================================="
echo "           Setup Complete!                "
echo "=========================================="
echo "To activate manually, run: source $VENV_DIR/bin/activate"
