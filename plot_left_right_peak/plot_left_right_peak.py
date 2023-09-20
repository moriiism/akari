#
# plot_pca_cat.py
#
# add pca components
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
#   % python $akari_tool/preproc/add_flag_pca.py
#   % python $akari_tool/preproc/plot_left_right_peak.py
#

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import mixture

# import seaborn as sns # visualize

from akarilib import get_colname_lst_of_pixarr_norm

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat_pca.csv"
data_df = pd.read_csv(incsv)
print(data_df)


data_left_df = data_df[(data_df["left"] == 1) & 
                       (data_df["dark"] == 0) &
                       (data_df["tzl_y"] > 5)]
data_left_df.reset_index(inplace=True, drop=True)
data_right_df = data_df[(data_df["left"] == 0) & 
                        (data_df["dark"] == 0) &
                        (data_df["tzl_y"] > 5)]
data_right_df.reset_index(inplace=True, drop=True)
print("left", len(data_left_df))
print("right", len(data_right_df))

data_left_cat_df = data_left_df[(data_left_df["nstar_cat8"] > 0)]
data_left_cat_df.reset_index(inplace=True, drop=True)
print("left_cat", len(data_left_cat_df))

data_left_pm1pix_df = data_left_df[(data_left_df["star_pos"] > 1)]
data_left_pm1pix_df.reset_index(inplace=True, drop=True)
print("left_pm1pix", len(data_left_pm1pix_df))

data_left_pm2pix_df = data_left_df[(data_left_df["star_pos"] > 0)]
data_left_pm2pix_df.reset_index(inplace=True, drop=True)
print("left_pm2pix", len(data_left_pm2pix_df))

data_left_noright_df = data_left_df[(data_left_df["star_pos"] == 0)]
data_left_noright_df.reset_index(inplace=True, drop=True)
print("left_noright", len(data_left_noright_df))

# print(data_left_noright_df["file_find"]) ---> all NaN

data_left_pm2pix_cluster_df = data_left_pm2pix_df[
    (data_left_pm2pix_df["pc01"] > 4.0) & 
    (data_left_pm2pix_df["pc02"] < 5.0)]
data_left_pm2pix_cluster_df.reset_index(inplace=True, drop=True)
print("left_pm2pix_cluster", len(data_left_pm2pix_cluster_df))

# all data 
peak_left_lst = []
peak_right_lst = []
peak_ratio_left_right_lst = []
peak_diff_left_right_lst = []
file_left_lst = []
file_right_lst = []
tzl_x_left_lst = []
tzl_x_right_lst = []
tzl_y_left_lst = []
tzl_y_right_lst = []
nstar_cat8_lst = []

# search corresponding right image
# loop for left hand side
for irow1 in range(len(data_left_df)):
    file_find = ""
    peak_left = 0.0
    tzl_x_left = 0
    tzl_y_left = 0
    nstar_cat8 = 0
    if (0 < data_left_df.loc[irow1, "nfind"]):
        file_find = data_left_df.loc[irow1, "file_find"]
        peak_left = data_left_df.loc[irow1, "x02y02"]
        tzl_x_left = data_left_df.loc[irow1, "tzl_x"]
        tzl_y_left = data_left_df.loc[irow1, "tzl_y"]
        nstar_cat8 = data_left_df.loc[irow1, "nstar_cat8"]
    else:
        continue
    
    # loop for right hand side
    for irow2 in range(len(data_right_df)):
        if(file_find == data_right_df.loc[irow2, "file"]):
            peak_right = data_right_df.loc[irow2, "x02y02"]
            tzl_x_right = data_right_df.loc[irow2, "tzl_x"]
            tzl_y_right = data_right_df.loc[irow2, "tzl_y"]
            peak_ratio_left_right = peak_right / peak_left
            peak_diff_left_right = peak_right - peak_left
            peak_left_lst.append(peak_left)
            peak_right_lst.append(peak_right)
            peak_ratio_left_right_lst.append(peak_ratio_left_right)
            peak_diff_left_right_lst.append(peak_diff_left_right)
            tzl_x_left_lst.append(tzl_x_left)
            tzl_x_right_lst.append(tzl_x_right)
            tzl_y_left_lst.append(tzl_y_left)
            tzl_y_right_lst.append(tzl_y_right)
            file_left_lst.append(data_left_df.loc[irow1, "file"])
            file_right_lst.append(data_right_df.loc[irow2, "file"])
            nstar_cat8_lst.append(nstar_cat8)
            break

print("left ", len(peak_left_lst))
print("right ", len(peak_right_lst))


####
# data_left_cat_df 
peak_left_cat_lst = []
peak_right_cat_lst = []
peak_ratio_left_right_cat_lst = []
peak_diff_left_right_cat_lst = []
file_left_cat_lst = []
file_right_cat_lst = []
tzl_x_left_cat_lst = []
tzl_x_right_cat_lst = []
tzl_y_left_cat_lst = []
tzl_y_right_cat_lst = []
nstar_cat8_cat_lst = []

# search corresponding right image
# loop for left hand side
for irow1 in range(len(data_left_cat_df)):
    file_find = ""
    peak_left = 0.0
    tzl_x_left = 0
    tzl_y_left = 0
    nstar_cat8 = 0    
    if (0 < data_left_cat_df.loc[irow1, "nfind"]):
        file_find = data_left_cat_df.loc[irow1, "file_find"]
        peak_left = data_left_cat_df.loc[irow1, "x02y02"]
        tzl_x_left = data_left_cat_df.loc[irow1, "tzl_x"]
        tzl_y_left = data_left_cat_df.loc[irow1, "tzl_y"]
        nstar_cat8 = data_left_cat_df.loc[irow1, "nstar_cat8"]
    else:
        continue
    
    # loop for right hand side
    for irow2 in range(len(data_right_df)):
        if(file_find == data_right_df.loc[irow2, "file"]):
            peak_right = data_right_df.loc[irow2, "x02y02"]
            tzl_x_right = data_right_df.loc[irow2, "tzl_x"]
            tzl_y_right = data_right_df.loc[irow2, "tzl_y"]
            peak_ratio_left_right = peak_right / peak_left
            peak_diff_left_right = peak_right - peak_left
            peak_left_cat_lst.append(peak_left)
            peak_right_cat_lst.append(peak_right)
            peak_ratio_left_right_cat_lst.append(peak_ratio_left_right)
            peak_diff_left_right_cat_lst.append(peak_diff_left_right)
            tzl_x_left_cat_lst.append(tzl_x_left)
            tzl_x_right_cat_lst.append(tzl_x_right)
            tzl_y_left_cat_lst.append(tzl_y_left)
            tzl_y_right_cat_lst.append(tzl_y_right)
            file_left_cat_lst.append(data_left_cat_df.loc[irow1, "file"])
            file_right_cat_lst.append(data_right_df.loc[irow2, "file"])
            nstar_cat8_cat_lst.append(nstar_cat8)
            break

print("left ", len(peak_left_cat_lst))
print("right ", len(peak_right_cat_lst))

############
# data_left_pm1pix_df 
peak_left_pm1pix_lst = []
peak_right_pm1pix_lst = []
peak_ratio_left_right_pm1pix_lst = []
peak_diff_left_right_pm1pix_lst = []
file_left_pm1pix_lst = []
file_right_pm1pix_lst = []
tzl_x_left_pm1pix_lst = []
tzl_x_right_pm1pix_lst = []
tzl_y_left_pm1pix_lst = []
tzl_y_right_pm1pix_lst = []
nstar_cat8_pm1pix_lst = []

# search corresponding right image
# loop for left hand side
for irow1 in range(len(data_left_pm1pix_df)):
    file_find = ""
    peak_left = 0.0
    tzl_x_left = 0
    tzl_y_left = 0
    nstar_cat8 = 0
    if (0 < data_left_pm1pix_df.loc[irow1, "nfind"]):
        file_find = data_left_pm1pix_df.loc[irow1, "file_find"]
        peak_left = data_left_pm1pix_df.loc[irow1, "x02y02"]
        tzl_x_left = data_left_pm1pix_df.loc[irow1, "tzl_x"]
        tzl_y_left = data_left_pm1pix_df.loc[irow1, "tzl_y"]
        nstar_cat8 = data_left_pm1pix_df.loc[irow1, "nstar_cat8"]
    else:
        continue
    
    # loop for right hand side
    for irow2 in range(len(data_right_df)):
        if(file_find == data_right_df.loc[irow2, "file"]):
            peak_right = data_right_df.loc[irow2, "x02y02"]
            tzl_x_right = data_right_df.loc[irow2, "tzl_x"]
            tzl_y_right = data_right_df.loc[irow2, "tzl_y"]
            peak_ratio_left_right = peak_right / peak_left
            peak_diff_left_right = peak_right - peak_left
            peak_left_pm1pix_lst.append(peak_left)
            peak_right_pm1pix_lst.append(peak_right)
            peak_ratio_left_right_pm1pix_lst.append(peak_ratio_left_right)
            peak_diff_left_right_pm1pix_lst.append(peak_diff_left_right)
            tzl_x_left_pm1pix_lst.append(tzl_x_left)
            tzl_x_right_pm1pix_lst.append(tzl_x_right)
            tzl_y_left_pm1pix_lst.append(tzl_y_left)
            tzl_y_right_pm1pix_lst.append(tzl_y_right)
            file_left_pm1pix_lst.append(data_left_pm1pix_df.loc[irow1, "file"])
            file_right_pm1pix_lst.append(data_right_df.loc[irow2, "file"])
            nstar_cat8_pm1pix_lst.append(nstar_cat8)
            break

print("left ", len(peak_left_pm1pix_lst))
print("right ", len(peak_right_pm1pix_lst))


############
# data_left_pm2pix_df 
peak_left_pm2pix_lst = []
peak_right_pm2pix_lst = []
peak_ratio_left_right_pm2pix_lst = []
peak_diff_left_right_pm2pix_lst = []
file_left_pm2pix_lst = []
file_right_pm2pix_lst = []
tzl_x_left_pm2pix_lst = []
tzl_x_right_pm2pix_lst = []
tzl_y_left_pm2pix_lst = []
tzl_y_right_pm2pix_lst = []
nstar_cat8_pm2pix_lst = []

# search corresponding right image
# loop for left hand side
for irow1 in range(len(data_left_pm2pix_df)):
    file_find = ""
    peak_left = 0.0
    tzl_x_left = 0
    tzl_y_left = 0
    nstar_cat8 = 0
    if (0 < data_left_pm2pix_df.loc[irow1, "nfind"]):
        file_find = data_left_pm2pix_df.loc[irow1, "file_find"]
        peak_left = data_left_pm2pix_df.loc[irow1, "x02y02"]
        tzl_x_left = data_left_pm2pix_df.loc[irow1, "tzl_x"]
        tzl_y_left = data_left_pm2pix_df.loc[irow1, "tzl_y"]
        nstar_cat8 = data_left_pm2pix_df.loc[irow1, "nstar_cat8"]
    else:
        continue
    
    # loop for right hand side
    for irow2 in range(len(data_right_df)):
        if(file_find == data_right_df.loc[irow2, "file"]):
            peak_right = data_right_df.loc[irow2, "x02y02"]
            tzl_x_right = data_right_df.loc[irow2, "tzl_x"]
            tzl_y_right = data_right_df.loc[irow2, "tzl_y"]
            peak_ratio_left_right = peak_right / peak_left
            peak_diff_left_right = peak_right - peak_left
            peak_left_pm2pix_lst.append(peak_left)
            peak_right_pm2pix_lst.append(peak_right)
            peak_ratio_left_right_pm2pix_lst.append(peak_ratio_left_right)
            peak_diff_left_right_pm2pix_lst.append(peak_diff_left_right)
            tzl_x_left_pm2pix_lst.append(tzl_x_left)
            tzl_x_right_pm2pix_lst.append(tzl_x_right)
            tzl_y_left_pm2pix_lst.append(tzl_y_left)
            tzl_y_right_pm2pix_lst.append(tzl_y_right)
            file_left_pm2pix_lst.append(data_left_pm2pix_df.loc[irow1, "file"])
            file_right_pm2pix_lst.append(data_right_df.loc[irow2, "file"])
            nstar_cat8_pm2pix_lst.append(nstar_cat8)
            break

print("left ", len(peak_left_pm2pix_lst))
print("right ", len(peak_right_pm2pix_lst))


############
# data_left_pm2pix_cluster_df 
peak_left_pm2pix_cluster_lst = []
peak_right_pm2pix_cluster_lst = []
peak_ratio_left_right_pm2pix_cluster_lst = []
peak_diff_left_right_pm2pix_cluster_lst = []
file_left_pm2pix_cluster_lst = []
file_right_pm2pix_cluster_lst = []
tzl_x_left_pm2pix_cluster_lst = []
tzl_x_right_pm2pix_cluster_lst = []
tzl_y_left_pm2pix_cluster_lst = []
tzl_y_right_pm2pix_cluster_lst = []
nstar_cat8_pm2pix_cluster_lst = []

# search corresponding right image
# loop for left hand side
for irow1 in range(len(data_left_pm2pix_cluster_df)):
    file_find = ""
    peak_left = 0.0
    tzl_x_left = 0
    tzl_y_left = 0
    nstar_cat8 = 0
    if (0 < data_left_pm2pix_cluster_df.loc[irow1, "nfind"]):
        file_find = data_left_pm2pix_cluster_df.loc[irow1, "file_find"]
        peak_left = data_left_pm2pix_cluster_df.loc[irow1, "x02y02"]
        tzl_x_left = data_left_pm2pix_cluster_df.loc[irow1, "tzl_x"]
        tzl_y_left = data_left_pm2pix_cluster_df.loc[irow1, "tzl_y"]
        nstar_cat8 = data_left_pm2pix_cluster_df.loc[irow1, "nstar_cat8"]
    else:
        continue
    
    # loop for right hand side
    for irow2 in range(len(data_right_df)):
        if(file_find == data_right_df.loc[irow2, "file"]):
            peak_right = data_right_df.loc[irow2, "x02y02"]
            tzl_x_right = data_right_df.loc[irow2, "tzl_x"]
            tzl_y_right = data_right_df.loc[irow2, "tzl_y"]
            peak_ratio_left_right = peak_right / peak_left
            peak_diff_left_right = peak_right - peak_left
            peak_left_pm2pix_cluster_lst.append(peak_left)
            peak_right_pm2pix_cluster_lst.append(peak_right)
            peak_ratio_left_right_pm2pix_cluster_lst.append(peak_ratio_left_right)
            peak_diff_left_right_pm2pix_cluster_lst.append(peak_diff_left_right)
            tzl_x_left_pm2pix_cluster_lst.append(tzl_x_left)
            tzl_x_right_pm2pix_cluster_lst.append(tzl_x_right)
            tzl_y_left_pm2pix_cluster_lst.append(tzl_y_left)
            tzl_y_right_pm2pix_cluster_lst.append(tzl_y_right)
            file_left_pm2pix_cluster_lst.append(
                data_left_pm2pix_cluster_df.loc[irow1, "file"])
            file_right_pm2pix_cluster_lst.append(
                data_right_df.loc[irow2, "file"])
            nstar_cat8_pm2pix_cluster_lst.append(nstar_cat8)
            break

print("left ", len(peak_left_pm2pix_cluster_lst))
print("right ", len(peak_right_pm2pix_cluster_lst))


# dump file list
print("dump file list ...")

outdir = indir
outfile_full = outdir + "/" + "peak_left_right_file.list"
print("outfile = ", outfile_full)
with open(outfile_full, "w") as fptr:
    for iline in range(len(file_left_lst)):
        print(file_left_lst[iline], 
              tzl_x_left_lst[iline],
              tzl_y_left_lst[iline],
              peak_left_lst[iline],
              file_right_lst[iline], 
              tzl_x_right_lst[iline],
              tzl_y_right_lst[iline],
              peak_right_lst[iline],
              nstar_cat8_lst[iline],
              file=fptr)

outdir = indir
outfile_full = outdir + "/" + "peak_left_right_file_cat.list"
print("outfile = ", outfile_full)
with open(outfile_full, "w") as fptr:
    for iline in range(len(file_left_cat_lst)):
        print(file_left_cat_lst[iline], 
              tzl_x_left_cat_lst[iline],
              tzl_y_left_cat_lst[iline],
              peak_left_cat_lst[iline],
              file_right_cat_lst[iline],
              tzl_x_right_cat_lst[iline],
              tzl_y_right_cat_lst[iline],
              peak_right_cat_lst[iline],
              nstar_cat8_cat_lst[iline],
              file=fptr)

outdir = indir
outfile_full = outdir + "/" + "peak_left_right_file_pm1pix.list"
print("outfile = ", outfile_full)
with open(outfile_full, "w") as fptr:
    for iline in range(len(file_left_pm1pix_lst)):
        print(file_left_pm1pix_lst[iline],
              tzl_x_left_pm1pix_lst[iline],
              tzl_y_left_pm1pix_lst[iline],
              peak_left_pm1pix_lst[iline],
              file_right_pm1pix_lst[iline],
              tzl_x_right_pm1pix_lst[iline],
              tzl_y_right_pm1pix_lst[iline],
              peak_right_pm1pix_lst[iline],
              nstar_cat8_pm1pix_lst[iline],
              file=fptr)

outdir = indir
outfile_full = outdir + "/" + "peak_left_right_file_pm2pix.list"
print("outfile = ", outfile_full)
with open(outfile_full, "w") as fptr:
    for iline in range(len(file_left_pm2pix_lst)):
        print(file_left_pm2pix_lst[iline],
              tzl_x_left_pm2pix_lst[iline],
              tzl_y_left_pm2pix_lst[iline],
              peak_left_pm2pix_lst[iline],
              file_right_pm2pix_lst[iline], 
              tzl_x_right_pm2pix_lst[iline],
              tzl_y_right_pm2pix_lst[iline],
              peak_right_pm2pix_lst[iline],
              nstar_cat8_pm2pix_lst[iline],
              file=fptr)


outdir = indir
outfile_full = outdir + "/" + "peak_left_right_file_pm2pix_cluster.list"
print("outfile = ", outfile_full)
with open(outfile_full, "w") as fptr:
    for iline in range(len(file_left_pm2pix_cluster_lst)):
        print(file_left_pm2pix_cluster_lst[iline],
              tzl_x_left_pm2pix_cluster_lst[iline],
              tzl_y_left_pm2pix_cluster_lst[iline],
              peak_left_pm2pix_cluster_lst[iline],
              file_right_pm2pix_cluster_lst[iline], 
              tzl_x_right_pm2pix_cluster_lst[iline],
              tzl_y_right_pm2pix_cluster_lst[iline],
              peak_right_pm2pix_cluster_lst[iline],
              nstar_cat8_pm2pix_cluster_lst[iline],
              file=fptr)


fig, ax = plt.subplots(2,2)

ax[0,0].scatter(peak_left_lst, peak_right_lst,
                s=1, c="black")
ax[0,0].scatter(peak_left_pm2pix_lst, peak_right_pm2pix_lst,
                s=1, c="blue")
ax[0,0].scatter(peak_left_pm1pix_lst, peak_right_pm1pix_lst,
                s=1, c="green")
ax[0,0].scatter(peak_left_cat_lst, peak_right_cat_lst,
                s=1, c="red")
ax[0,0].set_xlim(0.0, 10000.0)
ax[0,0].set_ylim(0.0, 10000.0)
ax[0,0].grid(True, linestyle='--')
#ax[0,0].set_xlabel("left")
ax[0,0].set_ylabel("right")


ax[0,1].scatter(peak_left_lst, peak_right_lst,
                s=1, c="black")
ax[0,1].scatter(peak_left_pm2pix_lst, peak_right_pm2pix_lst,
                s=1, c="blue")
ax[0,1].scatter(peak_left_pm1pix_lst, peak_right_pm1pix_lst,
                s=1, c="green")
ax[0,1].scatter(peak_left_cat_lst, peak_right_cat_lst,
                s=1, c="red")
ax[0,1].set_xlim(0.0, 1000.0)
ax[0,1].set_ylim(0.0, 1000.0)
ax[0,1].grid(True, linestyle='--')
#ax[0,1].set_xlabel("left")
#ax[0,1].set_ylabel("right")

ax[1,0].scatter(peak_left_lst, peak_right_lst,
                s=1, c="black")
ax[1,0].scatter(peak_left_pm2pix_lst, peak_right_pm2pix_lst,
                s=1, c="blue")
ax[1,0].scatter(peak_left_pm1pix_lst, peak_right_pm1pix_lst,
                s=1, c="green")
ax[1,0].scatter(peak_left_cat_lst, peak_right_cat_lst,
                s=1, c="red")
ax[1,0].set_xlim(0.0, 100.0)
ax[1,0].set_ylim(0.0, 100.0)
ax[1,0].grid(True, linestyle='--')
ax[1,0].set_xlabel("left")
ax[1,0].set_ylabel("right")


ax[1,1].scatter(peak_left_lst, peak_right_lst,
                s=1, c="black")
ax[1,1].scatter(peak_left_pm2pix_lst, peak_right_pm2pix_lst,
                s=1, c="blue")
ax[1,1].scatter(peak_left_pm1pix_lst, peak_right_pm1pix_lst,
                s=1, c="green")
ax[1,1].scatter(peak_left_cat_lst, peak_right_cat_lst,
                s=1, c="red")
ax[1,1].set_xlim(0.0, 10.0)
ax[1,1].set_ylim(0.0, 10.0)
ax[1,1].grid(True, linestyle='--')
ax[1,1].set_xlabel("left")
#ax[1,1].set_ylabel("right")

outdir = indir
outfile_full = outdir + "/" + "peak_left_right.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

#######################################

# log scale

fig, ax = plt.subplots(1)

plt.scatter(peak_left_lst, peak_right_lst,
                s=1, c="black")
plt.scatter(peak_left_pm2pix_lst, peak_right_pm2pix_lst,
                s=1, c="blue")
plt.scatter(peak_left_pm1pix_lst, peak_right_pm1pix_lst,
                s=1, c="green")
plt.scatter(peak_left_cat_lst, peak_right_cat_lst,
                s=1, c="red")
plt.grid(True, linestyle='--')
plt.xlabel("left")
plt.ylabel("right")
plt.xscale("log")
plt.yscale("log")
plt.title("peak")

outdir = indir
outfile_full = outdir + "/" + "peak_left_right_log.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


####################################
# right / left plot

fig, ax = plt.subplots(2)

ax[0].scatter(peak_left_lst, peak_ratio_left_right_lst,
            s=1, c="black")
ax[0].scatter(peak_left_pm2pix_lst, peak_ratio_left_right_pm2pix_lst,
            s=1, c="blue")
ax[0].scatter(peak_left_pm1pix_lst, peak_ratio_left_right_pm1pix_lst,
            s=1, c="green")
ax[0].scatter(peak_left_cat_lst, peak_ratio_left_right_cat_lst,
            s=1, c="red")
ax[0].grid(True, linestyle='--')
ax[0].set_xlabel("left")
ax[0].set_ylabel("right / left")
ax[0].set_xscale("log")
ax[0].set_title("peak")


ax[1].scatter(peak_left_lst, peak_ratio_left_right_lst,
            s=1, c="black")
ax[1].scatter(peak_left_pm2pix_lst, peak_ratio_left_right_pm2pix_lst,
            s=1, c="blue")
ax[1].scatter(peak_left_pm1pix_lst, peak_ratio_left_right_pm1pix_lst,
            s=1, c="green")
ax[1].scatter(peak_left_cat_lst, peak_ratio_left_right_cat_lst,
            s=1, c="red")
ax[1].grid(True, linestyle='--')
ax[1].set_xlabel("left")
ax[1].set_ylabel("right / left")
ax[1].set_xscale("log")
ax[1].set_ylim(0.0, 5.0)
ax[1].set_title("peak")

outdir = indir
outfile_full = outdir + "/" + "peak_ratio_left_right_log.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


####################################
# right - left plot

plt.scatter(peak_left_lst, peak_diff_left_right_lst,
            s=1, c="black")
plt.scatter(peak_left_pm2pix_lst, peak_diff_left_right_pm2pix_lst,
            s=1, c="blue")
plt.scatter(peak_left_pm1pix_lst, peak_diff_left_right_pm1pix_lst,
            s=1, c="green")
plt.scatter(peak_left_cat_lst, peak_diff_left_right_cat_lst,
            s=1, c="red")
plt.grid(True, linestyle='--')
plt.xlabel("left")
plt.ylabel("right - left")
plt.xscale("log")
# plt.yscale("log")
plt.title("peak")

outdir = indir
outfile_full = outdir + "/" + "peak_diff_left_right_log.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


###############################

peak_ratio_left_right_cat_arr = np.array(peak_ratio_left_right_cat_lst)
print("mean(peak_ratio_left_right_cat) = ", 
      np.mean(peak_ratio_left_right_cat_arr))
print("std(peak_ratio_left_right_cat) = ",
      np.std(peak_ratio_left_right_cat_arr))
