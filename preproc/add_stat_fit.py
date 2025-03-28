#
# add_stat_fit.py
#
# add statistical values by gaussian fits
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
#

import os
import sys
import pandas as pd
from akarilib import calc_2dgaussfit_in_row_of_dataframe
from akarilib import get_colname_lst_of_pixarr_norm
from akarilib import get_meshgrid_xybin_center
import numpy as np

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat.csv"
data_df = pd.read_csv(incsv)
print(data_df)

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
data_pixarr_norm_df = pd.read_csv(
    incsv, usecols=colname_pixarr_norm_lst)
data_pixarr_norm_df["index"] = data_pixarr_norm_df.index

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('Agg')

#data_1darr = data_df.flatten()

# make mesh
nbinx = 5
nbiny = 5
lo_xbin_center = -1.0
up_xbin_center = 1.0
lo_ybin_center = -1.0
up_ybin_center = 1.0
(xbin_center_2darr, ybin_center_2darr) = get_meshgrid_xybin_center(
    nbinx, lo_xbin_center, up_xbin_center,
    nbiny, lo_ybin_center, up_ybin_center)

fit_stat_df = data_pixarr_norm_df.apply(
    calc_2dgaussfit_in_row_of_dataframe,
    args=(xbin_center_2darr, ybin_center_2darr), axis=1)

data_add_df = pd.concat([data_df,
                         fit_stat_df], axis=1)
print(data_add_df)
print(data_add_df.columns)

print("gfit_valid:")
print(data_add_df["gfit_valid"].value_counts())

outdir = indir
outcsv = outdir + "/" + "akari_stat_fit.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)

