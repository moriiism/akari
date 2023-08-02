#
# check_cat_dist.py
#
# check catalog star distance
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
#   % python $akari_tool/preproc/check_cat_dist.py
#

import os
import sys
import time
import math
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, angular_separation, Angle
from astropy import units
import matplotlib.pyplot as plt

from akarilib import calc_angular_separation_in_row_of_dataframe

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat.csv"
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
dist_df = pd.DataFrame([], index=range(nrow),
                       columns=["dist_cat",
                                "ra_cat_nearest",
                                "dec_cat_nearest"])
dist_df.loc[:, "dist_cat"] = -1.0
dist_df.loc[:, "ra_cat_nearest"] = -1.0
dist_df.loc[:, "dec_cat_nearest"] = -100.0

# count star_pos
print(data_sel_df["star_pos"].value_counts())

xlo_dist_hist = 0.0
xup_dist_hist = 20.0
bins_dist_hist = 20
dist_hist, dist_hist_bin_edges = np.histogram(
    [], bins=bins_dist_hist, 
    range=(xlo_dist_hist, xup_dist_hist))

cat_dist_lst = []

for irow1 in range(len(data_sel_df)):
    ra = data_sel_df.loc[irow1, "ra"]
    dec = data_sel_df.loc[irow1, "dec"]
    star_pos = data_sel_df.loc[irow1, "star_pos"]
    if(star_pos < 1):
    # if(star_pos < 2):
        continue

    print(irow1, star_pos, ra, dec)
    time_st = time.time()
    cat_sel_df = cat_df[ (cat_df.ra_cat < 1.0) |
                         (cat_df.ra_cat > 359.0) |
                         (cat_df.dec_cat < -89.0) |
                         (cat_df.dec_cat > 89.0) |
                         ( (abs(ra - cat_df.ra_cat) < 1.0) &
                           (abs(dec - cat_df.dec_cat) < 1.0) ) ]
    #print(len(cat_sel_df))
    #print(cat_sel_df)

    ra_cat_nearest = -1.0
    dec_cat_nearest = -100.0
    dist_cat = 3600.0
    for irow2, (ra_cat, dec_cat) in enumerate(zip(
            cat_sel_df["ra_cat"].values,
            cat_sel_df["dec_cat"].values)):
        separation = angular_separation(
            Angle(ra, units.degree),
            Angle(dec, units.degree),
            Angle(ra_cat, units.degree),
            Angle(dec_cat, units.degree))
        sep_arcsec = separation.to(units.arcsec).value
        if (sep_arcsec < dist_cat):
            dist_cat = sep_arcsec
            ra_cat_nearest = ra_cat
            dec_cat_nearest = dec_cat
        if (sep_arcsec < 3600.0):
            cat_dist_lst.append(sep_arcsec)

    dist_df.loc[irow1, "dist_cat"] = dist_cat
    dist_df.loc[irow1, "ra_cat_nearest"] = ra_cat_nearest
    dist_df.loc[irow1, "dec_cat_nearest"] = dec_cat_nearest

    time_ed = time.time()
    # print(time_ed - time_st)

print(dist_df)
data_add_df = pd.concat([data_df, dist_df], axis=1)
print(data_add_df)

outdir = indir
outcsv = outdir + "/" + "dist_tmp.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)


# fill cat_dist
fig, ax = plt.subplots(3,1)
#ax.set_xlim(0.0, 20.0)
#ax.set_ylim(0.0, 10.0)
ax[0].hist(cat_dist_lst, bins=50, range=[0.0, 2000.0])
ax[1].hist(cat_dist_lst, bins=50, range=[0.0, 200.0])
ax[2].hist(cat_dist_lst, bins=50, range=[0.0, 20.0])

plt.xlabel("dist_cat")
plt.ylabel('frequency')
#ax.set_xscale(e)
#ax.set_yscale(yscale)
#plt.xlim(0.0, 20.0)
#plt.ylim(0.0, 10.0)

outfile_full = outdir + "/" + "cat_dist.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()
del fig
del ax
