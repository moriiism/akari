#!/bin/sh

export PYTHONPATH=/home/morii/work/github/moriiism/akari
export AKARI_TOOL_DIR=/home/morii/work/github/moriiism/akari/spectrum
export AKARI_DATA_DIR=/home/morii/work/akari/data/spectrum
export AKARI_ANA_DIR=/home/morii/work/akari/ana/spectrum

echo "PYTHONPATH="$PYTHONPATH
echo "AKARI_TOOL_DIR="$AKARI_TOOL_DIR
echo "AKARI_DATA_DIR="$AKARI_DATA_DIR
echo "AKARI_ANA_DIR="$AKARI_ANA_DIR
termtitle="spectrum"
PROMPT_COMMAND='echo -ne "\033]0;${termtitle}\007"'
echo "setup done."
