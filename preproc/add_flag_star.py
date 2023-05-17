#
# addflag.py
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#
# add flag of star/spike
#

import os
import pandas as pd

from util import getColNameLst
from util import calcNormInRowOfDataFrame, calcStatInRowOfDataFrame

indir = "/home/morii/work/akari/ana/spikethumb_20230407"
incsv = indir + "/" + "akari_stat.csv"
data_df = pd.read_csv(incsv)
print(data_df)
nrow = len(data_df)

data_sel_df = data_df[["file", "tzl_x", "tzl_y"]]
flag_df = pd.DataFrame([], index=range(nrow), columns=["star"])
flag_df.loc[:, "star"] = 0
flag_df["star"] = flag_df["star"].astype(int)

for irow1 in range(len(data_sel_df)):
    file1 = data_sel_df.loc[irow1, "file"]
    tzl_x_1 = data_sel_df.loc[irow1, "tzl_x"]
    tzl_y_1 = data_sel_df.loc[irow1, "tzl_y"]
    file1_split = file1.split(".")
    (obsid1, dummy, sernum1) = file1_split[0].split("_")
    for irow2 in range(len(data_sel_df)):
        file2 = data_sel_df.loc[irow2, "file"]
        tzl_x_2 = data_sel_df.loc[irow2, "tzl_x"]
        tzl_y_2 = data_sel_df.loc[irow2, "tzl_y"]
        file2_split = file2.split(".")
        (obsid2, dummy, sernum2) = file2_split[0].split("_")
        if( (obsid1 == obsid2) and
            (sernum1 != sernum2) and
            ( abs(tzl_x_1 - tzl_x_2) < 3 ) and
            ( abs(tzl_y_1 - tzl_y_2) < 3 ) ):
            print(file1, file2, tzl_x_1, tzl_x_2, tzl_y_1, tzl_y_2)
            flag_df.loc[irow1, "star"] = 1

print(flag_df)
data_add_df = pd.concat([data_df, flag_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat_flag.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
