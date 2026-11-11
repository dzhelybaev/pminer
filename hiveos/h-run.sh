#!/usr/bin/env bash
cd "$(dirname "$0")"
. h-manifest.conf
mkdir -p /run/pminer "$(dirname "$CUSTOM_LOG_BASENAME")"
ARGS=$(cat "$CUSTOM_CONFIG_FILENAME" 2>/dev/null)
./pminer --stats /run/pminer/stats.json $ARGS 2>&1 | tee --append "${CUSTOM_LOG_BASENAME}.log"
