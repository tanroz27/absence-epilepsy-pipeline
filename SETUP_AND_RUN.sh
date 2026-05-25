#!/bin/bash
set -e
echo ""
echo "════════════════════════════════════════════"
echo "  Rozendal Thesis Pipeline"
echo "════════════════════════════════════════════"
echo ""
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from https://www.python.org"
    exit 1
fi
echo "Python: $(python3 --version)"
echo ""
echo "Installing packages..."
python3 -m pip install --quiet --upgrade numpy scipy matplotlib h5py
echo "Packages OK."
echo ""
DATA_DIR="/Volumes/home/Dennis_to_Kevin/LP"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data drive not found at $DATA_DIR"
    echo "Plug in the drive and re-run."
    exit 1
fi
echo "Data drive: found"
echo ""
cd "$(dirname "$0")"
echo "Running pipeline..."
python3 run_all_recordings.py
echo ""
echo "════════════════════════════════════════════"
echo "  Done. Figures saved to:"
echo "  $(pwd)/output_figures/"
echo "════════════════════════════════════════════"
