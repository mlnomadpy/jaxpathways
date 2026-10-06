#!/bin/sh
# Run from the repository or extracted project workspace root on a Docker-capable CI worker.
set -eu
python3 projects/engineering-release/tests/check.py --implementation solution --stage all
engineering_tmp=$(mktemp -d)
trap 'rm -rf "$engineering_tmp"' EXIT HUP INT TERM
python3 projects/engineering-release/mlflow_lab.py --output "$engineering_tmp/run"
python3 projects/engineering-release/container_check.py --model "$engineering_tmp/run/model.json" --output "$engineering_tmp/container-report.json"
# Copy this receipt into the CI system's artifact store before the temporary workspace is cleaned.
cat "$engineering_tmp/container-report.json"
