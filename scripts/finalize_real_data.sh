#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW="$ROOT/data/raw"
REAL="$ROOT/data/processed_real"
ZIP="$RAW/CGMacros_dateshifted365.zip"
PART="$ZIP.part"
URL="https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"

command -v python >/dev/null || { echo "Python is required."; exit 1; }
command -v curl >/dev/null || { echo "curl is required."; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required."; exit 1; }

# Keep local research dependencies isolated from the Mac's system Python.
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python -m venv "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt

mkdir -p "$RAW"

# Download safely into a .part file. If the connection drops, curl resumes
# from the existing partial file instead of restarting the ~627 MB archive.
if [ -f "$ZIP" ]; then
  echo "CGMacros archive already downloaded; verifying it..."
  if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
    echo "Archive is incomplete/corrupt; moving it to resumable partial download."
    mv "$ZIP" "$PART"
  fi
fi

if [ ! -f "$ZIP" ]; then
  echo "Downloading CGMacros v1.0.0 (~627 MB)."
  echo "If the connection resets, rerun this same script; it will resume from the partial file."

  # --continue-at - resumes an interrupted .part file.
  # --retry-all-errors also retries connection resets and transient network errors.
  curl -L --fail --continue-at - \
    --retry 12 --retry-delay 5 --retry-max-time 3600 --retry-all-errors \
    --connect-timeout 30 --progress-bar \
    "$URL" -o "$PART"

  echo "Verifying downloaded archive..."
  if ! unzip -tq "$PART" >/dev/null 2>&1; then
    echo "Download is still incomplete or invalid. The partial file has been kept at:"
    echo "  $PART"
    echo "Run this script again to resume the download."
    exit 1
  fi

  mv "$PART" "$ZIP"
  echo "CGMacros archive verified successfully."
else
  echo "CGMacros archive verified successfully."
fi

if [ ! -f "$RAW/.extracted" ]; then
  echo "Extracting CGMacros..."
  rm -rf "$RAW/extracted"
  mkdir -p "$RAW/extracted"
  unzip -q -o "$ZIP" -d "$RAW/extracted"
  touch "$RAW/.extracted"
fi

rm -rf "$REAL"
mkdir -p "$REAL"

"$PY" scripts/prepare_cgmacros.py \
  --raw-root "$RAW/extracted" \
  --output-root "$REAL"

echo "Running leakage-safe real-data evaluation..."
METATWIN_DATA_ROOT="$REAL" "$PY" scripts/evaluate.py
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
