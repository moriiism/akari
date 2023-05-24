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
#   % python $akari_tool/preproc/add_flag_star_catalog.py
#

import os
import sys
import time
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation, Angle
from astropy import units

from akarilib import calc_angular_separation_in_row_of_dataframe

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_star.csv"
data_df = pd.read_csv(incsv)
print(data_df)
nrow = len(data_df)

datadir = os.environ["AKARI_DATA_DIR"]
cat_table = datadir + "/../" + "AKARI-IRC_PSC_V1.txt"
cat_df = pd.read_table(cat_table, sep='\s+', header=None, usecols=[2,3])
cat_df.columns = ["ra_cat", "dec_cat"]
print(cat_df)
nrow_cat = len(cat_df)

data_sel_df = data_df[["ra", "dec"]]
flag_df = pd.DataFrame([], index=range(nrow), columns=["star_cat"])
flag_df.loc[:, "star_cat"] = 0
flag_df["star_cat"] = flag_df["star_cat"].astype(int)

for irow1 in range(len(data_sel_df)):
    print(irow1)
    ra = data_sel_df.loc[irow1, "ra"]
    dec = data_sel_df.loc[irow1, "dec"]
    time_st = time.time()
    print(time_st)
    for irow2, (ra_cat, dec_cat) in enumerate(zip(
            cat_df["ra_cat"].values,
            cat_df["dec_cat"].values)):
        separation = angular_separation(
            Angle(ra, units.degree),
            Angle(dec, units.degree),
            Angle(ra_cat, units.degree),
            Angle(dec_cat, units.degree))
        if (irow2 % 100000 == 0):
            print(irow2)
        if (separation.to(units.arcsec).value < 5):
            flag_df.iloc[irow1, "star_cat"] = 1
            next
    time_ed = time.time()
    print(time_ed - time_st)

exit()

        #match_df = cat_df.apply(
        #calc_angular_separation_in_row_of_dataframe,
        #args=(ra, dec), axis=1)
        #sum = match_df.sum()
        #print(sum)


print(flag_df)
data_add_df = pd.concat([data_df, flag_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat_star_cat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
