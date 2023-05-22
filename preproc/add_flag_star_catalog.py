#
# add_flag_star.py
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
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation, Angle
from astropy import units

from akarilib import getColNameLst
from akarilib import calcNormInRowOfDataFrame, calcStatInRowOfDataFrame

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
flag_df = pd.DataFrame([], index=range(nrow), columns=["cat"])
flag_df.loc[:, "cat"] = 0
flag_df["cat"] = flag_df["cat"].astype(int)

for irow1 in range(len(data_sel_df)):
    print(irow1)
    ra = data_sel_df.loc[irow1, "ra"]
    dec = data_sel_df.loc[irow1, "dec"]
    for irow2 in range(len(cat_df)):
        ra_cat = cat_df.loc[irow2, "ra_cat"]
        dec_cat = cat_df.loc[irow2, "dec_cat"]
        separation = angular_separation(
            Angle(ra, units.degree),
            Angle(dec, units.degree),
            Angle(ra_cat, units.degree),
            Angle(dec_cat, units.degree))
        if (separation.to(units.arcsec).value < 5):
            flag_df.loc[irow1, "cat"] = 1
            print(separation.to(units.arcsec))

print(flag_df)
data_add_df = pd.concat([data_df, flag_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "akari_stat_star_cat.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
