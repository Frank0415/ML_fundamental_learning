#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/project"

cd "$PROJECT_DIR"
source /home/xc/Documents/learning_ML/11-868/llmsys_hw2/.venv/bin/activate 
python run_sentiment.py "$@"
