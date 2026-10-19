#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")" && pwd)
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    echo 'Run bash setup.sh first.' >&2
    exit 1
fi
cd "$project_root"
exec "$project_root/.venv/bin/python" "$@"
