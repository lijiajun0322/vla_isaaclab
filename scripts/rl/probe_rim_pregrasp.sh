#!/bin/bash
set -e
source "$(dirname "$0")/../activate.sh"
check_install_environment
cd "$VLA_ISAACLAB_PROJECT"
exec python scripts/rl/probe_rim_pregrasp.py "$@"
