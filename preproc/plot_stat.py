#
# plot_stat.py
#
# plot stat
#

import os
import sys
import pandas as pd
from akarilib import get_colname_lst_of_pixarr
from akarilib import calc_norm_in_row_of_dataframe
from akarilib import calc_stat_for_normed_in_row_of_dataframe
from akarilib import calc_feature_for_normed_in_row_of_dataframe
import matplotlib.pyplot as plt

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat.csv"
data_df = pd.read_csv(incsv)
print(data_df)
print(data_df.columns)

data_selrow_df = data_df[(data_df["left"] == 1) & 
                         (data_df["dark"] == 0) &
                         (data_df["edge"] == 0) &
                         (data_df["star_pos"] > 1)]
print(len(data_selrow_df))
data_selrow_cat_df = data_df[(data_df["left"] == 1) & 
                             (data_df["dark"] == 0) &
                             (data_df["edge"] == 0) &
                             (data_df["star_pos"] > 1) &
                             (data_df["nstar_cat8"] > 0)]
print(len(data_selrow_cat_df))


# hist
val_xlim_list = [
    ["tz_x", -1, -1, 0],
    ["tz_y", -1, -1, 0],
    ["tzl_x", -1, -1, 0],
    ["tzl_y", -1, -1, 0],
    ["ti", -1, -1, 0],
    ["id", -1, -1, 0],
    ["af_tim", -1, -1, 0],
    ["sum", -1, -1, 1],
    ["ave_around", 0, 100, 0],
    ["ave_margin", 0, 50, 0],
    ["norm_stddev", 0, 0.05, 0],
    ["norm_min", -0.02, 0.05, 0],
    ["norm_max", 0, 0.3, 0],
    ["norm_skew", -2, 5, 0],
    ["norm_kurt", -5, 25, 0],
    ["norm_gini", 0, 1, 0],
    ["ratio_around_to_peak", 0, 1, 0],
    ["norm_ave_margin", 0.0, 0.05, 0],
    ["gfit_mu_x", -1, 1, 0],
    ["gfit_mu_y", -1, 1, 0],
    ["gfit_sigma_x", 0, 0.5, 0],
    ["gfit_sigma_y", 0, 0.5, 0],
    ["gfit_theta", 0, 1.5, 0],
    ["gfit_norm", 0, 2, 0],
    ["gfit_const", 0, 0.04, 0],
    ["gfit_valid", 0, 2, 0],
    ["nfind", 0, 6, 0],
    ["x02y02", -1, -1, 1],
    ["x02y02_norm", -1, -1, 1]]
# "crval1", "crval2", "ra", "dec",

outdir_this = os.environ["AKARI_ANA_DIR"] + "/" + "stat"
if (False == os.path.exists(outdir_this)):
    os.makedirs(outdir_this)

data_id = os.path.basename(os.environ["AKARI_ANA_DIR"])

fig, ax = plt.subplots(1,1)    
for (val, xlo, xup, ylog) in val_xlim_list:
    # data_df[val].hist(bins=20)
    if (xlo < xup):
        plt.hist(data_selrow_df[val], bins=100, range=[xlo, xup])
        plt.hist(data_selrow_cat_df[val], bins=100, range=[xlo, xup])
    else:
        plt.hist(data_selrow_df[val], bins=100)
        plt.hist(data_selrow_cat_df[val], bins=100)
    plt.xlabel(val)
    plt.ylabel("frequency")
    plt.grid()
    plt.title(data_id)
    #plt.ylim(0,50)
    if (xlo < xup):
        plt.xlim(xlo, xup)
    if (ylog == 1):
        plt.yscale('log')

    outfile_full = outdir_this + "/" + ("%s.png" % (val))
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()


# 2d plot
plt.scatter(data_selrow_df["sum"], data_selrow_df["x02y02"])
plt.scatter(data_selrow_cat_df["sum"], data_selrow_cat_df["x02y02"])
plt.xlabel("sum")
plt.ylabel("x02y02")
plt.grid()
plt.title(data_id)
outfile_full = outdir_this + "/" + "sum_x02y02.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


plt.close()
