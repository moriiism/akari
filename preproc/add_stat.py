#
# addstat.py
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#
# add statistical values
#

import os
import pandas as pd

from util import getColNameLst
from util import calcNormInRowOfDataFrame, calcStatInRowOfDataFrame

indir = "/home/morii/work/akari/ana/spikethumb_20230407"
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
print(data_df)

data_sel_df = data_df.drop(["file", "tzl_x", "tzl_y"], axis=1)
data_norm_df = data_sel_df.apply(calcNormInRowOfDataFrame, axis=1)
print(data_norm_df)
data_stat_df = data_norm_df.apply(calcStatInRowOfDataFrame, axis=1)
print(data_stat_df)
data_add_df = pd.concat([data_df, data_norm_df, data_stat_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
