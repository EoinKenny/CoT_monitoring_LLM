#!/bin/bash
set -euo pipefail
/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python -u prepare_download.py
/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python -u smoke.py
