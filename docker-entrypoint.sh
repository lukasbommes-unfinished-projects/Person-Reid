#!/bin/sh
set -e

# Make endernewton object detector
make -C sfmt/endernewton

# Create directories and files
mkdir -p logs

exec "$@"
