#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW="$ROOT/data/raw"
REAL="$ROOT/data/processed_real"
ZIP="$RAW/CGMacros_dateshifted365.zip"
URL="https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"

command -v python >/dev/null || { echo "Python is required."; exit 1; }
command -v curl >/dev/null || { echo "curl is required."; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required."; exit 1; }

mkdir -p "$RAW"

if [ ! -f "$ZIP" ]; then
  echo "Downloading CGMacros v1.0.0 (~627 MB)..."
  curl -L --fail --retry 3 --progress-bar "$URL" -o "$ZIP"
else
  echo "CGMacros archive already exists; reusing it."
fi

if [ ! -f "$RAW/.extracted" ]; then
  echo "Extracting CGMacros..."
  unzip -q -o "$ZIP" -d "$RAW/extracted"
  touch "$RAW/.extracted"
fi

rm -rf "$REAL"
mkdir -p "$REAL"

python scripts/prepare_cgmacros.py \
  --raw-root "$RAW/extracted" \
  --output-root "$REAL"

echo "Running leakage-safe real-data evaluation..."
METATWIN_DATA_ROOT="$REAL" python scripts/evaluate.py

echo "Real-data research tables are now in results/tables/."

echo "Raw CGMacros and processed_real data are intentionally not committed."

# Only derived research outputs are published; no raw or participant-level data.
git add results/tables scripts src README.md .gitignore
git commit -m "Update MetaTwin research results from CGMacros" || true
git push origin main

echo "Deploying the updated API..."
if command -v railway >/dev/null; then
  railway up
else
  echo "Railway CLI not found; run 'railway up' manually."
fi

echo "Done."
