#
# add_flag_star_catalog.py
#
# add flag of star by catalog
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
#   % python $akari_tool/preproc/add_flag_star_pos.py

#   % python $akari_tool/preproc/add_flag_star_catalog.py
#

import os
import sys
import time
import math
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation, Angle
from astropy import units

from akarilib import calc_angular_separation_in_row_of_dataframe

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_pca.csv"
data_df = pd.read_csv(incsv)
print(data_df)
nrow = len(data_df)

datadir = os.environ["AKARI_DATA_DIR"]
cat_table = datadir + "/../" + "AKARI-IRC_PSC_V1.txt"
cat_df = pd.read_table(cat_table, sep='\s+', 
                       header=None, usecols=[2,3])
cat_df.columns = ["ra_cat", "dec_cat"]
print(cat_df)
nrow_cat = len(cat_df)

data_sel_df = data_df[["ra", "dec", "star_pos"]]
flag_df = pd.DataFrame([], index=range(nrow),
                       columns=["nstar_cat8"])
flag_df.loc[:, "nstar_cat8"] = 0
flag_df["nstar_cat8"] = flag_df["nstar_cat8"].astype(int)

# count star_pos
print(data_sel_df["star_pos"].value_counts())

for irow1 in range(len(data_sel_df)):
    ra = data_sel_df.loc[irow1, "ra"]
    dec = data_sel_df.loc[irow1, "dec"]
    star_pos = data_sel_df.loc[irow1, "star_pos"]

    # select within +-1 pixel
    if(star_pos < 2):
        continue

    print(star_pos)
    print(irow1, ra, dec)

    time_st = time.time()
    cat_sel_df = cat_df[ (cat_df.ra_cat < 1.0) |
                         (cat_df.ra_cat > 359.0) |
                         (cat_df.dec_cat < -89.0) |
                         (cat_df.dec_cat > 89.0) |
                         ( (abs(ra - cat_df.ra_cat) < 1.0) &
                           (abs(dec - cat_df.dec_cat) < 1.0) ) ]
    #print(len(cat_sel_df))
    #print(cat_sel_df)

    for irow2, (ra_cat, dec_cat) in enumerate(zip(
            cat_sel_df["ra_cat"].values,
            cat_sel_df["dec_cat"].values)):
        separation = angular_separation(
            Angle(ra, units.degree),
            Angle(dec, units.degree),
            Angle(ra_cat, units.degree),
            Angle(dec_cat, units.degree))
        if (separation.to(units.arcsec).value <= 8):
            flag_df.loc[irow1, "nstar_cat8"] += 1
            print(irow1, irow2, "find8")
            print(ra, dec, ra_cat, dec_cat)


    time_ed = time.time()
    # print(time_ed - time_st)

print(flag_df)
data_add_df = pd.concat([data_df, flag_df], axis=1)
print(data_add_df)

# count nstar_cat8
print("nstar_cat8")
print(data_add_df["nstar_cat8"].value_counts())

# count star_pos
print("star_pos")
print(data_add_df["star_pos"].value_counts())

# count nstar_cat8, star_pos
print("nstar_cat8, star_pos")
print(data_add_df[["nstar_cat8", "star_pos"]].value_counts())

# count nstar_cat8, star_pos, nfind, diff_tzl_x, diff_tzl_y
print("nstar_cat8, star_pos, nfind, diff_tzl_x, diff_tzl_y")
print(data_add_df[["nstar_cat8", "star_pos", "nfind",
                   "diff_tzl_x", "diff_tzl_y"]].value_counts())


outdir = indir
outcsv = outdir + "/" + "akari_stat_fit_star_pca_cat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
