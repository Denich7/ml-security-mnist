#!/usr/bin/env bash
# Local equivalent of the workflow. Stop at the first failed mandatory check.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-ci/bin/activate
export PATH="$PWD/.tools:$PATH"
mkdir -p reports/lab3
date -u +%FT%TZ > reports/lab3/checked-at.txt
git rev-parse HEAD > reports/lab3/commit.txt
python --version > reports/lab3/python-version.txt
docker --version > reports/lab3/docker-version.txt
trivy --version > reports/lab3/trivy-version.txt
python -m src.ci prepare
ruff check src/ tests/ --output-format json --output-file reports/lab3/ruff.json
python -m bandit -r src/ -f json -o reports/lab3/bandit.json
python -m pytest tests/ -v --junitxml=reports/lab3/pytest.xml | tee reports/lab3/pytest.txt
python -m pip_audit -r requirements-ci.lock -f json -o reports/lab3/pip-audit.json
python -m src.ci data
python -m src.ci evaluate
docker build --tag ml-project:lab3 . 2>&1 | tee reports/lab3/docker-build.txt
trivy config --exit-code 1 --format json --output reports/lab3/trivy-config.json Dockerfile
# Native scan fails for CRITICAL/UNKNOWN, and the report validator is an extra gate.
trivy image --scanners vuln --severity UNKNOWN,CRITICAL --exit-code 1 --format json \
  --output reports/lab3/trivy-image.json ml-project:lab3
python -m src.ci security
uid=$(docker run --rm --entrypoint id ml-project:lab3 -u)
printf '%s\n' "$uid" | tee reports/lab3/container-uid.txt
test "$uid" != 0
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 64 --memory 256m --cpus 1 \
  ml-project:lab3 | tee reports/lab3/smoke.txt
sha256sum models/mnist_linear.npz > reports/lab3/model.sha256
echo 'ALL CHECKS PASSED'
