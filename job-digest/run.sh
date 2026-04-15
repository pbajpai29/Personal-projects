#!/bin/bash
# Climate Finance Job Digest — runner
# Called by launchd every 48 hours.

cd "$(dirname "$0")"

LOG_FILE="$(dirname "$0")/digest.log"
PYTHON="/Users/pbajpai/anaconda3/bin/python3"

echo "------------------------------------------------------------" >> "$LOG_FILE"
echo "Run started: $(date)" >> "$LOG_FILE"

"$PYTHON" main.py >> "$LOG_FILE" 2>&1
EXIT=$?

echo "Run finished: $(date) (exit $EXIT)" >> "$LOG_FILE"
