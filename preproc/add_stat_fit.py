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
import numpy as np

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat.csv"
data_df = pd.read_csv(incsv)

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
data_pixarr_norm_df = pd.read_csv(
    incsv, usecols=colname_pixarr_norm_lst)


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
xbin_center_1darr = np.linspace(lo_xbin_center,
                                up_xbin_center,
                                nbinx)
ybin_center_1darr = np.linspace(lo_ybin_center,
                                up_ybin_center,
                                nbiny)
(xbin_center_2darr, ybin_center_2darr) = np.meshgrid(
    xbin_center_1darr, ybin_center_1darr)

print(data_df)

fit_stat_df = data_pixarr_norm_df.apply(
    calc_2dgaussfit_in_row_of_dataframe,
    args=(xbin_center_2darr, ybin_center_2darr), axis=1)

data_add_df = pd.concat([data_df,
                         fit_stat_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat_fit.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)

