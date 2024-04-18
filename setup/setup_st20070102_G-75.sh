#!/bin/sh

branch_name=develop
data_name=st20070102_G-75
model_name=st20070102_G-75

export AKARI_DATA_DIR=/home/morii/work/akari/data/${data_name}
export AKARI_TOOL_DIR=/home/morii/work/github/moriiism/akari
export PYTHONPATH=$AKARI_TOOL_DIR
export AKARI_ANA_DIR=/home/morii/work/akari/ana/${branch_name}/${data_name}
export MODEL_DIR=/home/morii/work/akari/ana/${branch_name}/${model_name}

echo "AKARI_DATA_DIR="$AKARI_DATA_DIR
echo "AKARI_TOOL_DIR="$AKARI_TOOL_DIR
echo "PYTHONPATH="$PYTHONPATH
echo "AKARI_ANA_DIR="$AKARI_ANA_DIR
echo "MODEL_DIR="$MODEL_DIR

termtitle="BRANCH=${branch_name}, DATA=${data_name}, MODEL=${model_name}"
PROMPT_COMMAND='echo -ne "\033]0;${termtitle}\007"'
echo "setup done."
