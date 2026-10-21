#!/bin/bash
set -euo pipefail
/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python -u download_curl.py
/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python -u smoke.py
