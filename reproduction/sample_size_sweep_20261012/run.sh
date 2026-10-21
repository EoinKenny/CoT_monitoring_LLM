#!/bin/bash
set -euo pipefail
for sample_size in 200 300 500; do
 OUT="/home/ubuntu/experiments/sample-stability-n${sample_size}-20261012" SAMPLE_N="$sample_size" REPS=10000 /home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python /home/ubuntu/experiments/bootstrap_size_sweep_20261012.py
done
