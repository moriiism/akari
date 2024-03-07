#!/bin/bash

cd /home/morii/work/github/moriiism/akari/preproc

source ../setup/setup_st20070102A2.sh
python3 plot_stat.py
source ../setup/setup_stm13r.sh
python3 plot_stat.py
source ../setup/setup_stm43r.sh
python3 plot_stat.py
source ../setup/setup_stm63r.sh
python3 plot_stat.py


