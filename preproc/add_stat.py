#
# add_stat.py
#
# add statistical values
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#
# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool/preproc/add_stat.py
#

import os
import sys
import pandas as pd
from akarilib import calc_norm_in_row_of_dataframe
from akarilib import calc_stat_in_row_of_dataframe
from akarilib import calc_feature_in_row_of_dataframe


indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
print(data_df)

data_sel_df = data_df.drop(["file", "tzl_x", "tzl_y",
                            "crval1", "crval2", "ra", "dec"], axis=1)
data_norm_df = data_sel_df.apply(
    calc_norm_in_row_of_dataframe, axis=1)
print(data_norm_df)
data_stat_df = data_norm_df.apply(
    calc_stat_in_row_of_dataframe, axis=1)
print(data_stat_df)
data_feature_df = data_norm_df.apply(
    calc_feature_in_row_of_dataframe, axis=1)

data_add_df = pd.concat([data_df,
                         data_norm_df,
                         data_stat_df,
                         data_feature_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
