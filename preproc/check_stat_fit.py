#
# check_stat_fit.py
#
# check gaussian fit
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
#   % python $akari_tool/preproc/check_stat_fit.py

import os
import sys
import time
import math
import pandas as pd
from akarilib import check_2dgaussfit_in_row_of_dataframe
from akarilib import get_meshgrid_xybin_center
from akarilib import get_colname_lst_of_pixarr_norm

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit.csv"

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
colname_lst = colname_pixarr_norm_lst + ["gfit_mu_x",
                                         "gfit_mu_y",
                                         "gfit_sigma_x",
                                         "gfit_sigma_y",
                                         "gfit_theta",
                                         "gfit_norm",
                                         "gfit_const"]
data_df = pd.read_csv(incsv, usecols=colname_lst)

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

data_df.apply(
    check_2dgaussfit_in_row_of_dataframe,
    args=(xbin_center_2darr, ybin_center_2darr), axis=1)





