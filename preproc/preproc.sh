#!/bin/sh

# data with (ra, dec) info

python3 fits_to_csv.py 1
# --> akari.csv

python3 plot_ti.py

# python3 split_csv_by_frameid.py

python3 add_stat.py
# --> akari_stat.csv

python3 add_stat_fit.py
# --> akari_stat_fit.csv

python3 add_flag_star_pos.py
# --> akari_stat_fit_star.csv

python3 add_flag_star_catalog.py
# --> akari_stat_fit_star_cat.csv
