#!/bin/sh

branch=develop

export PYTHONPATH=/home/morii/work/github/moriiism/akari
export AKARI_TOOL_DIR=/home/morii/work/github/moriiism/akari
export AKARI_ANA_DIR=/home/morii/work/akari/ana/${branch}/stm43r_3

echo "PYTHONPATH="$PYTHONPATH
echo "AKARI_TOOL_DIR="$AKARI_TOOL_DIR
echo "AKARI_ANA_DIR="$AKARI_ANA_DIR

termtitle="stm43r_3"
PROMPT_COMMAND='echo -ne "\033]0;${termtitle}\007"'
echo "setup done."
