#!/bin/sh

# data with no (ra, dec) info

python3 fits_to_csv.py 0
python3 plot_ti.py
python3 add_stat.py
python3 add_stat_fit.py
python3 add_flag_star_pos.py

