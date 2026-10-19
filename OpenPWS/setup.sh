#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bootstrap="${PYTHON_BOOTSTRAP:-python3.12}"
case "${1:-gpu}" in
    gpu) requirements=requirements.txt ;;
    cpu) requirements=requirements-cpu.txt ;;
    *) echo 'Usage: bash setup.sh [gpu|cpu]' >&2; exit 2 ;;
esac
if [[ ! -f .venv/pyvenv.cfg ]]; then "$bootstrap" -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
fi
.venv/bin/python -m pip install -r "$requirements"
.venv/bin/python -m pip check
