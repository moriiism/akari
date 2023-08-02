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




# count nstar_cat5
print("nstar_cat5")
print(data_df["nstar_cat5"].value_counts())

# count nstar_cat10
print("nstar_cat10")
print(data_df["nstar_cat10"].value_counts())



# count star_pos
print("star_pos")
print(data_df["star_pos"].value_counts())

# count nfind
print("nfind")
print(data_df["nfind"].value_counts())
print(data_df[["nstar_cat5", "nfind"]].value_counts())


print("nstar_cat5, star_pos")
print(data_df[["nstar_cat5", "star_pos"]].value_counts())

print("nstar_cat10, star_pos")
print(data_df[["nstar_cat10", "star_pos"]].value_counts())


print("nstar_cat5, star_pos, diff_tzl_x, diff_tzl_y")
print(data_df[["nstar_cat5", "star_pos",
               "diff_tzl_x", "diff_tzl_y"]].value_counts())

print((data_df[data_df["nstar_cat5"]>=1]))
data_sel_df = data_df[data_df["nstar_cat5"]>=1]

print(data_sel_df[["file", "file_find"]])



# count star_pos
print("star_pos")
print(data_df["star_pos"].value_counts())





