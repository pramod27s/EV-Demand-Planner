#!/usr/bin/env bash
# Full ML pipeline (Git Bash / Linux / macOS). Usage: scripts/run_all.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the project root
PY="${PYTHON:-python}"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

$PY -m src.data.preprocess
$PY -m src.experiments.run_benchmarks
$PY -m src.experiments.run_baselines
$PY -m src.experiments.run_tuning "$@"
$PY -m src.experiments.summarize
$PY -m src.experiments.export_artifacts
$PY -m pytest -q
