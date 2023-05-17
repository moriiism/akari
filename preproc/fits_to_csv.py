#
# fits_to_csv.py
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib

# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool/preproc/fits_to_csv.py
#

import os
import sys
from astropy.io import fits
import pandas as pd
from akarilib import getColNameLst

# keyword

# TZL_X/TZL_Y
colname_lst = ["file", "tzl_x", "tzl_y"] + getColNameLst() 
data_all_df = pd.DataFrame([], columns=colname_lst)
data_all_df = data_all_df.astype(float)
data_all_df['file'] = data_all_df['file'].astype(str)
data_all_df['tzl_x'] = data_all_df['tzl_x'].astype(int)
data_all_df['tzl_y'] = data_all_df['tzl_y'].astype(int)

# file = "/home/morii/work/akari/data/spikethumb_20230407/
# F0870074916_4NS_S091.fits.gz"

data_dir = os.environ["AKARI_DATA_DIR"]
data_lst = os.listdir(data_dir)
ndata = len(data_lst)
idata = 0
for dataname in data_lst:
    print(f"idata = {idata}({ndata}): {dataname}")
    hdu = fits.open(data_dir + "/" + dataname)
    hdu0 = hdu[0]
    data_1d = hdu0.data.flatten()
    data_df = pd.DataFrame([], columns=colname_lst)
    data_df = data_df.astype(float)
    data_df['file'] = data_df['file'].astype(str)
    data_df.loc[0] = 0
    data_df.iloc[0,0] = dataname
    data_df.iloc[0,1] = hdu0.header["TZL_X"]
    data_df.iloc[0,2] = hdu0.header["TZL_Y"]
    data_df.iloc[0,3:] = data_1d.tolist()
    data_all_df = pd.concat([data_all_df, data_df], ignore_index=True)

    del data_df
    del data_1d
    hdu.close()
    idata += 1

print(data_all_df)

outdir = os.environ["AKARI_ANA_DIR"]
outcsv = outdir + "/" + "akari.csv"
print(f"outcsv = {outcsv}")
data_all_df.to_csv(outcsv, index=False)

