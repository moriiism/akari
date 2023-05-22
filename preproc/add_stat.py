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
from akarilib import getColNameLst
from akarilib import calcNormInRowOfDataFrame, calcStatInRowOfDataFrame

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
print(data_df)

data_sel_df = data_df.drop(["file", "tzl_x", "tzl_y",
                            "crval1", "crval2", "ra", "dec"], axis=1)
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
