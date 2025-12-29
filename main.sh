#!/usr/bin/env bash

set -euo pipefail

# CURL_CA_BUNDLE= \
# SSL_CERT_FILE= \
# REQUESTS_CA_BUNDLE= \

# Examples:
#   1) Start the normal scheduler/server (default behavior):
#      ./main.sh
#
#   2) Run one-shot inference and compare approaches (writes per-variant outputs under output/<active_set>/):
#      ./main.sh --run-once-inference --active-set transfer --inference-variants event,grid:1s
#
# Preprocessing env vars (used by the scheduled jobs):
#   DATA_APPROACH=grid|event
#   TIME_GRID_FREQ=1s|30s|1min|...

if command -v poetry >/dev/null 2>&1; then
	PYTHONPATH=./src \
	poetry run python ./src/ml_monitoring_service/main.py "$@"
else
	PYTHONPATH=./src \
	python ./src/ml_monitoring_service/main.py "$@"
fi
