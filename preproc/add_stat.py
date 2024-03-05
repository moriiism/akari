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
#   % python $akari_tool/preproc/fits_to_csv.py
#   % python $akari_tool/preproc/add_stat.py
#

import os
import sys
import pandas as pd
from akarilib import get_colname_lst_of_pixarr
from akarilib import calc_norm_in_row_of_dataframe

from akarilib import calc_feature_for_asis_in_row_of_dataframe
from akarilib import calc_feature_for_normed_in_row_of_dataframe
from akarilib import calc_stat_for_normed_in_row_of_dataframe

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
print(data_df.columns)
print(data_df)

colname_lst = get_colname_lst_of_pixarr()
data_sel_df = data_df[colname_lst]
print(data_sel_df.columns)

data_norm_df = data_sel_df.apply(
    calc_norm_in_row_of_dataframe, axis=1)
print(data_norm_df.columns)

# for asis data
data_asis_feature_df = data_sel_df.apply(
    calc_feature_for_asis_in_row_of_dataframe, axis=1)
print(data_asis_feature_df.columns)

# for normed data
data_normed_stat_df = data_norm_df.apply(
    calc_stat_for_normed_in_row_of_dataframe, axis=1)
print(data_normed_stat_df.columns)

data_normed_feature_df = data_norm_df.apply(
    calc_feature_for_normed_in_row_of_dataframe, axis=1)
print(data_normed_feature_df.columns)

data_add_df = pd.concat([data_df,
                         data_norm_df,
                         data_asis_feature_df,
                         data_normed_stat_df,
                         data_normed_feature_df], axis=1)
print(data_add_df.columns)

outdir = indir
outcsv = outdir + "/" + "akari_stat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
