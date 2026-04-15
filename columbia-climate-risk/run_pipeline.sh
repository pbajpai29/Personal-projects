#!/usr/bin/env bash
# Run the full pipeline then launch the dashboard
set -e
cd "$(dirname "$0")"

echo "Step 1 — Weather data"
python 01_download_weather.py

echo "Step 2 — MTA data"
python 02_download_mta.py

echo "Step 3 — Merge & process"
python 03_merge_process.py

echo "Step 4 — ML analysis"
python 04_ml_analysis.py

echo ""
echo "Pipeline complete. Launching dashboard…"
streamlit run dashboard.py
