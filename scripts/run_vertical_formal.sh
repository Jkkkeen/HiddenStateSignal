#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m vertical.cli formal "$@"
