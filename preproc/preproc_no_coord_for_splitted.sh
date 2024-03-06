#!/bin/sh

# data with no (ra, dec) info
# data after splitted

python3 add_stat.py
python3 add_stat_fit.py
python3 add_flag_star_pos.py
