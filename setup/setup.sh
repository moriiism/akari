#!/bin/sh

export PYTHONPATH=/home/morii/work/github/moriiism/akari
export AKARI_TOOL_DIR=/home/morii/work/github/moriiism/akari
# export AKARI_DATA_DIR=/home/morii/work/akari/data/spikethumb_20230105
# export AKARI_DATA_DIR=/home/morii/work/akari/data/spikethumb_20230616
export AKARI_DATA_DIR=/home/morii/work/akari/data/spikethumb_20230616
export AKARI_ANA_DIR=/home/morii/work/akari/ana/spikethumb_20230616

echo "PYTHONPATH="$PYTHONPATH
echo "AKARI_TOOL_DIR="$AKARI_TOOL_DIR
echo "AKARI_DATA_DIR="$AKARI_DATA_DIR
echo "AKARI_ANA_DIR="$AKARI_ANA_DIR
echo "setup done."
