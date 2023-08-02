#!/bin/sh

export PYTHONPATH=/home/morii/work/github/moriiism/akari
export AKARI_TOOL_DIR=/home/morii/work/github/moriiism/akari
export AKARI_DATA_DIR=/home/morii/work/akari/data/st20070102B
export AKARI_ANA_DIR=/home/morii/work/akari/ana/st20070102B

echo "PYTHONPATH="$PYTHONPATH
echo "AKARI_TOOL_DIR="$AKARI_TOOL_DIR
echo "AKARI_DATA_DIR="$AKARI_DATA_DIR
echo "AKARI_ANA_DIR="$AKARI_ANA_DIR
termtitle="st20070102B"
PROMPT_COMMAND='echo -ne "\033]0;${termtitle}\007"'
echo "setup done."
