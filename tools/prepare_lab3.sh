#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(git branch --show-current)" = lab3 || { echo 'Switch to branch lab3 first'; exit 1; }
sha256sum -c reports/dataset.sha256
sha256sum -c metrics/model.sha256
if [[ -e ci/baseline.json ]]; then
  echo 'CI reference already exists. Do not initialize twice.'
  exit 1
fi
python3 -m venv .venv-ci
.venv-ci/bin/python -m pip install pip==26.3
.venv-ci/bin/python -m pip install -r requirements-ci.in
.venv-ci/bin/python -m pip freeze > requirements-ci.lock
.venv-ci/bin/python -m src.ci init
echo 'Now fixing the official Python image by digest...'
docker pull python:3.14-slim-trixie
image=$(docker image inspect python:3.14-slim-trixie --format '{{index .RepoDigests 0}}')
docker run --rm --entrypoint python "$image" -c 'import platform; print(platform.python_version())' > ci/python-version
{
  printf 'FROM %s\n' "$image"
  cat ci/Dockerfile.body
} > Dockerfile
.venv-ci/bin/python tools/install_trivy.py
echo 'Preparation complete. Add MODEL_KEY to GitHub Actions secrets before pushing.'
