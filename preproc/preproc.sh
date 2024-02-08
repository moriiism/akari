#!/bin/sh

python fits_to_csv.py 1
# --> akari.csv

python plot_ti.py
# python split_csv_by_frameid.py

python add_stat.py
# --> akari_stat.csv

python add_stat_fit.py
# --> akari_stat_fit.csv

python add_flag_star_pos.py
# --> akari_stat_fit_star.csv

python add_flag_star_catalog.py
# --> akari_stat_fit_star_cat.csv

python add_flag_pca.py 0
# --> akari_stat_fit_star_cat_pca.csv

python add_flag_pca_margin.py 0
# --> akari_stat_fit_star_cat_pca_margin.csv


# python add_flag_pca_v2.py
# python add_flag_pca_v3.py
# python check_cat_dist.py
# python check_star_catalog.py
# check_stat_fit.py
