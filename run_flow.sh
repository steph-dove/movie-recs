#!/bin/bash
# Wrapper script to run the Metaflow flow with the virtual environment
# Usage: ./run_flow.sh run --num_pages 5 --query_title "Arrival" --top_k 5
# Note: Use underscores (--num_pages) not dashes (--num-pages) for parameters

cd "$(dirname "$0")"
source .venv/bin/activate
python3 flow.py "$@"

