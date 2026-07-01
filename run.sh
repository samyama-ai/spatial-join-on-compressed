#!/usr/bin/env bash
# One-command reproduction. Regenerates every number + figure into results/.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt
echo "[1/3] experiments (synthetic + TIGER translate-sweep + cross-county)"
python -u src/experiment.py --synthetic \
    --tiger-fips 22051,27137,53033 \
    --tiger-pair "22051,22103;27031,27137" \
    --tiger-max 200 --out results
echo "[2/3] analysis vs pre-registered decision rules"
python src/analyze.py results
echo "[3/3] figures"
python src/make_figures.py results
echo "done. See results/verdict.json and results/figures/."
