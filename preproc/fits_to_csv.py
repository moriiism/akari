#
# fits_to_csv.py
#
# convert fits file data to csv file
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
#

import os
import sys
from astropy.io import fits
import pandas as pd
from akarilib import get_colname_lst_of_pixarr
from akarilib import ecliptic_to_radec

# keyword
# TZ_X/TZ_Y
# TZL_X/TZL_Y
#  left side  (1 <= TZ_X < 64) : TZL_X = TZ_X, TZL_Y = TZ_Y
#  right side (64 <= TZ_X)     : TZL_X = TZ_X - 64, TZL_Y = TZ_Y + 2
# CRVAL1/2 (Ecliptic coordinate system, Longitude/Lattitude)
colname_lst = (["file", "tz_x", "tz_y", "tzl_x", "tzl_y",
                "crval1", "crval2", "ra", "dec", "dark",
                "left", "edge"]
               + get_colname_lst_of_pixarr())
data_all_df = pd.DataFrame([], columns=colname_lst)
data_all_df = data_all_df.astype(float)
data_all_df['file'] = data_all_df['file'].astype(str)
data_all_df['tz_x'] = data_all_df['tz_x'].astype(int)
data_all_df['tz_y'] = data_all_df['tz_y'].astype(int)
data_all_df['tzl_x'] = data_all_df['tzl_x'].astype(int)
data_all_df['tzl_y'] = data_all_df['tzl_y'].astype(int)
data_all_df['dark'] = data_all_df['dark'].astype(int)
data_all_df['left'] = data_all_df['left'].astype(int)
data_all_df['edge'] = data_all_df['edge'].astype(int)

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
    data_1darr = hdu0.data.flatten()
    data_df = pd.DataFrame([], columns=colname_lst)
    data_df = data_df.astype(float)
    data_df['file'] = data_df['file'].astype(str)
    data_df.loc[0] = 0
    data_df.iloc[0,0] = dataname
    data_df.iloc[0,1] = hdu0.header["TZ_X"]
    data_df.iloc[0,2] = hdu0.header["TZ_Y"]
    data_df.iloc[0,3] = hdu0.header["TZL_X"]
    data_df.iloc[0,4] = hdu0.header["TZL_Y"]
    data_df.iloc[0,5] = hdu0.header["CRVAL1"]
    data_df.iloc[0,6] = hdu0.header["CRVAL2"]
    ecliptic_lon = hdu0.header["CRVAL1"]
    ecliptic_lat = hdu0.header["CRVAL2"]
    (ra, dec) = ecliptic_to_radec(ecliptic_lon, ecliptic_lat)
    data_df.iloc[0,7] = ra
    data_df.iloc[0,8] = dec
    # dark
    if (hdu0.header["TZL_X"] <= 6):
        data_df.iloc[0,9] = 1
    else:
        data_df.iloc[0,9] = 0

    # left
    if (hdu0.header["TZ_X"] < 64):
        data_df.iloc[0,10] = 1
    else:
        data_df.iloc[0,10] = 0

    # edge
    if (hdu0.header["TZL_Y"] <= 4):
        data_df.iloc[0,11] = 1
    else:
        data_df.iloc[0,11] = 0

    data_df.iloc[0,12:] = data_1darr.tolist()
    data_all_df = pd.concat([data_all_df, data_df], ignore_index=True)

    del data_df
    del data_1darr
    hdu.close()
    idata += 1

print(data_all_df)

print(data_all_df.columns)
print(data_all_df["dark"].value_counts())
print(data_all_df["left"].value_counts())
print(data_all_df["edge"].value_counts())

outdir = os.environ["AKARI_ANA_DIR"]
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

outcsv = outdir + "/" + "akari.csv"
print(f"outcsv = {outcsv}")
data_all_df.to_csv(outcsv, index=False)
