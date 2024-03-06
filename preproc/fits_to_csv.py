#
# fits_to_csv.py
#
# convert fits file data to csv file
#

# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool_dir/preproc/fits_to_csv.py
#

import os
import sys
from astropy.io import fits
import pandas as pd
from akarilib import get_colname_lst_of_pixarr
from akarilib import ecliptic_to_radec

flag_crval = 0
args = sys.argv
nargs = len(args) - 1
print(nargs)
if (1 == nargs):
    flag_crval = int(args[1])
    print("flag_crval = ", flag_crval)
    pass
else:
    print('usage: python fits_to_csv.py 1/0')
    print('usage: argument means that fits files contain crval(1) or not(0).')
    print('usage: CRVAL: coordinate info')
    print('Arguments are not 1.')
    exit()

# file = "/home/morii/work/akari/data/spikethumb_20230407/
# F0870074916_4NS_S091.fits.gz"

data_dir = os.environ["AKARI_DATA_DIR"]
data_lst = os.listdir(data_dir)
ndata = len(data_lst)
print("ndata = ", ndata)

# keyword
# TZ_X/TZ_Y
# TZL_X/TZL_Y
#  left side  (1 <= TZ_X < 64) : TZL_X = TZ_X, TZL_Y = TZ_Y
#  right side (64 <= TZ_X)     : TZL_X = TZ_X - 64, TZL_Y = TZ_Y + 2
# CRVAL1/2 (Ecliptic coordinate system, Longitude/Lattitude)

colname_lst = []
if (1 == flag_crval):
    colname_lst = (["file", "tz_x", "tz_y", "tzl_x", "tzl_y",
                    "ti", "id", "af_tim",
                    "crval1", "crval2", "ra", "dec", 
                    "dark", "left", "edge"]
                   + get_colname_lst_of_pixarr())
elif (0 == flag_crval):
    colname_lst = (["file", "tz_x", "tz_y", "tzl_x", "tzl_y",
                    "ti", "id", "af_tim",
                    "dark", "left", "edge"]
                   + get_colname_lst_of_pixarr())
else:
    print("bad flag_crval = ", flag_crval)
    exit()

data_all_df = pd.DataFrame(index=range(ndata), columns=colname_lst)
data_all_df.fillna(0, inplace=True)
data_all_df = data_all_df.astype(float)
data_all_df['file'] = data_all_df['file'].astype(str)
data_all_df['tz_x'] = data_all_df['tz_x'].astype(int)
data_all_df['tz_y'] = data_all_df['tz_y'].astype(int)
data_all_df['tzl_x'] = data_all_df['tzl_x'].astype(int)
data_all_df['tzl_y'] = data_all_df['tzl_y'].astype(int)
data_all_df['dark'] = data_all_df['dark'].astype(int)
data_all_df['left'] = data_all_df['left'].astype(int)
data_all_df['edge'] = data_all_df['edge'].astype(int)
data_all_df['ti']   = data_all_df['ti'].astype(int)
data_all_df['id']   = data_all_df['id'].astype(int)
data_all_df['af_tim']   = data_all_df['af_tim'].astype(int)

data_df = pd.DataFrame(index=range(1), columns=colname_lst)
data_df.fillna(0, inplace=True)
data_df = data_df.astype(float)
data_df['file'] = data_df['file'].astype(str)
data_df['tz_x'] = data_df['tz_x'].astype(int)
data_df['tz_y'] = data_df['tz_y'].astype(int)
data_df['tzl_x'] = data_df['tzl_x'].astype(int)
data_df['tzl_y'] = data_df['tzl_y'].astype(int)
data_df['dark'] = data_df['dark'].astype(int)
data_df['left'] = data_df['left'].astype(int)
data_df['edge'] = data_df['edge'].astype(int)
data_df['ti']   = data_df['ti'].astype(int)
data_df['id']   = data_df['id'].astype(int)
data_df['af_tim']   = data_df['af_tim'].astype(int)
data_df.loc[0] = 0

idata = 0
for dataname in data_lst:
    if (idata % 100 == 0):
        print(f"idata = {idata}({ndata}): {dataname}")
    hdu = fits.open(data_dir + "/" + dataname)
    hdu0 = hdu[0]
    data_1darr = hdu0.data.flatten()
    data_df.loc[0,"file"] = dataname
    data_df.loc[0,"tz_x"] = hdu0.header["TZ_X"]
    data_df.loc[0,"tz_y"] = hdu0.header["TZ_Y"]
    data_df.loc[0,"tzl_x"] = hdu0.header["TZL_X"]
    data_df.loc[0,"tzl_y"] = hdu0.header["TZL_Y"]
    data_df.loc[0,"ti"] = hdu0.header["CSDS_TI"]
    data_df.loc[0,"id"] = hdu0.header["FRAME_ID"]
    data_df.loc[0,"af_tim"] = hdu0.header["AF_TIM"]

    if (flag_crval == 1):
        data_df.loc[0,"crval1"] = hdu0.header["CRVAL1"]
        data_df.loc[0,"crval2"] = hdu0.header["CRVAL2"]

        ecliptic_lon = hdu0.header["CRVAL1"]
        ecliptic_lat = hdu0.header["CRVAL2"]
        (ra, dec) = ecliptic_to_radec(ecliptic_lon, ecliptic_lat)
        data_df.loc[0,"ra"] = ra
        data_df.loc[0,"dec"] = dec
    elif (flag_crval == 0):
        pass
    else:
        print("bad flag_crval = ", flag_crval)
        exit()

    # dark
    if (hdu0.header["TZL_X"] <= 7):
        data_df.loc[0,"dark"] = 1
    else:
        data_df.loc[0,"dark"] = 0

    # left
    if (hdu0.header["TZ_X"] < 64):
        data_df.loc[0,"left"] = 1
    else:
        data_df.loc[0,"left"] = 0

    # edge
    if (hdu0.header["TZ_Y"] <= 5):
        data_df.loc[0,"edge"] = 1
    else:
        data_df.loc[0,"edge"] = 0

    data_df.loc[0,get_colname_lst_of_pixarr()] = data_1darr.tolist()
    data_all_df.iloc[idata,:] = data_df.iloc[0,:]

    del data_1darr
    hdu.close()
    idata += 1

del data_df

print(">>> data_all_df:")
print(data_all_df)
print(">>>  data_all_df.loc[0,:):")
print(data_all_df.loc[0,:])
print(">>>  data_all_df.columns:")
print(data_all_df.columns)
print(">>>  data_all_df[\"dark\"].value_counts():")
print(data_all_df["dark"].value_counts())
print(">>>  data_all_df[\"left\"].value_counts():")
print(data_all_df["left"].value_counts())
print(">>> data_all_df[\"edge\"].value_counts():")
print(data_all_df["edge"].value_counts())

outdir = os.environ["AKARI_ANA_DIR"]
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

outcsv = outdir + "/" + "akari.csv"
print(f"outcsv = {outcsv}")
data_all_df.to_csv(outcsv, index=False)

del data_all_df
