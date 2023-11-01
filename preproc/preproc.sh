#!/bin/sh

python fits_to_csv.py 1
python plot_ti.py
# python split_csv_by_frameid.py
python add_stat.py
python add_stat_fit.py
python add_flag_star_pos.py
python add_flag_pca.py

# python add_flag_pca_v2.py
# python add_flag_pca_v3.py

python gmm_2d_2cl.py

python add_flag_star_catalog.py

# python check_cat_dist.py
# python check_star_catalog.py

# check_stat_fit.py
