#
# check_star_catalog.py
#
# check catalog star
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#
# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool/preproc/fits_to_csv.py
#   % python $akari_tool/preproc/add_stat.py
#   % python $akari_tool/preproc/add_stat_fit.py
#   % python $akari_tool/preproc/add_flag_star_pos.py
#   % python $akari_tool/preproc/add_flag_star_catalog.py
#   % python $akari_tool/preproc/check_star_catalog.py
#

import os
import sys
import time
import math
import pandas as pd

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat.csv"
data_df = pd.read_csv(incsv)
print(data_df)
nrow = len(data_df)

# count star_cat
print("star_cat")
print(data_df["star_cat"].value_counts())

# count star_pos
print("star_pos")
print(data_df["star_pos"].value_counts())

# count nfind
print("nfind")
print(data_df["nfind"].value_counts())

print("star_cat, star_pos")
print(data_df[["star_cat", "star_pos"]].value_counts())

print("star_cat, star_pos, diff_tzl_x, diff_tzl_y")
print(data_df[["star_cat", "star_pos",
               "diff_tzl_x", "diff_tzl_y"]].value_counts())
