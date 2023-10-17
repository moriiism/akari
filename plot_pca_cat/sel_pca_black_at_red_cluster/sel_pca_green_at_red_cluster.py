#
# sel_pca_green_at_red_cluster.py
#
# select cluster found by plot_pca_cat.py
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
#   % python $akari_tool/plot_pca_cat/plot_pca_cat.py
#   % python $akari_tool/plot_pca_cat/sel_pca_black_at_red_cluster/
#            sel_pca_green_at_red_cluster.py

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
from astropy.io import fits

# import seaborn as sns # visualize

from akarilib import get_colname_lst_of_pixarr_norm

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat_pca.csv"
data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["left"] == 1) &
                       (data_df["dark"] == 0) &
                       (data_df["tz_y"] > 5)]
data_right_df = data_df[(data_df["left"] == 0) &
                        (data_df["dark"] == 0) &
                        (data_df["tz_y"] > 5)]

data_left_cat_df = data_left_df[(data_left_df["nstar_cat8"] > 0)]
data_left_pm1pix_df = data_left_df[(data_left_df["star_pos"] > 1)]
data_left_pm2pix_df = data_left_df[(data_left_df["star_pos"] > 0)]
data_left_black_df = data_left_df[
    (data_left_df["star_pos"] == 0)]

data_right_cat_df = data_right_df[(data_right_df["nstar_cat8"] > 0)]
data_right_pm1pix_df = data_right_df[(data_right_df["star_pos"] > 1)]
data_right_pm2pix_df = data_right_df[(data_right_df["star_pos"] > 0)]
data_right_black_df = data_right_df[
    (data_right_df["star_pos"] == 0)]

### get min and max for scatter plot range

pc01_min = min(data_left_df['pc01'].min(), 
              data_right_df['pc01'].min())
pc01_max = max(data_left_df['pc01'].max(), 
              data_right_df['pc01'].max())
pc02_min = min(data_left_df['pc02'].min(), 
              data_right_df['pc02'].min())
pc02_max = max(data_left_df['pc02'].max(), 
              data_right_df['pc02'].max())
pc01_mid = (pc01_min + pc01_max) / 2.0
pc02_mid = (pc02_min + pc02_max) / 2.0
pc01_wid = (pc01_max - pc01_min) * 1.1
pc02_wid = (pc02_max - pc02_min) * 1.1
pc01_lo = pc01_mid - pc01_wid / 2.0
pc01_up = pc01_mid + pc01_wid / 2.0
pc02_lo = pc02_mid - pc02_wid / 2.0
pc02_up = pc02_mid + pc02_wid / 2.0

### left

### select cluster
# 0 -- 5
data_left_green_pc01_0to5_df = data_left_pm1pix_df[
    (data_left_pm1pix_df['pc01'] > 0.0) &
    (data_left_pm1pix_df['pc01'] < 5.0) &
    (data_left_pm1pix_df['pc02'] > -5.0) &
    (data_left_pm1pix_df['pc02'] < 5.0) & 
    (data_left_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_left_green_at_red_cluster_pc01_0to5.csv")
print(outfile_full)
data_left_green_pc01_0to5_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_left_green_at_red_cluster_pc01_0to5"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_left_green_pc01_0to5_df.index:
    file_name = data_left_green_pc01_0to5_df.loc[irow, "file"]
    tz_x = data_left_green_pc01_0to5_df.loc[irow, "tz_x"]
    tz_y = data_left_green_pc01_0to5_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


# 5 -- 10
data_left_green_pc01_5to10_df = data_left_pm1pix_df[
    (data_left_pm1pix_df['pc01'] > 5.0) &
    (data_left_pm1pix_df['pc01'] < 10.0) &
    (data_left_pm1pix_df['pc02'] > -5.0) &
    (data_left_pm1pix_df['pc02'] < 5.0) & 
    (data_left_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_left_green_at_red_cluster_pc01_5to10.csv")
print(outfile_full)
data_left_green_pc01_5to10_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_left_green_at_red_cluster_pc01_5to10"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_left_green_pc01_5to10_df.index:
    file_name = data_left_green_pc01_5to10_df.loc[irow, "file"]
    tz_x = data_left_green_pc01_5to10_df.loc[irow, "tz_x"]
    tz_y = data_left_green_pc01_5to10_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


# 10 -- 15
data_left_green_pc01_10to15_df = data_left_pm1pix_df[
    (data_left_pm1pix_df['pc01'] > 10.0) &
    (data_left_pm1pix_df['pc01'] < 15.0) &
    (data_left_pm1pix_df['pc02'] > -5.0) &
    (data_left_pm1pix_df['pc02'] < 5.0) & 
    (data_left_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_left_green_at_red_cluster_pc01_10to15.csv")
print(outfile_full)
data_left_green_pc01_10to15_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_left_green_at_red_cluster_pc01_10to15"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_left_green_pc01_10to15_df.index:
    file_name = data_left_green_pc01_10to15_df.loc[irow, "file"]
    tz_x = data_left_green_pc01_10to15_df.loc[irow, "tz_x"]
    tz_y = data_left_green_pc01_10to15_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


### right

### select cluster
# 0 -- 5
data_right_green_pc01_0to5_df = data_right_pm1pix_df[
    (data_right_pm1pix_df['pc01'] > 0.0) &
    (data_right_pm1pix_df['pc01'] < 5.0) &
    (data_right_pm1pix_df['pc02'] > -5.0) &
    (data_right_pm1pix_df['pc02'] < 5.0) & 
    (data_right_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_right_green_at_red_cluster_pc01_0to5.csv")
print(outfile_full)
data_right_green_pc01_0to5_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_right_green_at_red_cluster_pc01_0to5"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_right_green_pc01_0to5_df.index:
    file_name = data_right_green_pc01_0to5_df.loc[irow, "file"]
    tz_x = data_right_green_pc01_0to5_df.loc[irow, "tz_x"]
    tz_y = data_right_green_pc01_0to5_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


# 5 -- 10
data_right_green_pc01_5to10_df = data_right_pm1pix_df[
    (data_right_pm1pix_df['pc01'] > 5.0) &
    (data_right_pm1pix_df['pc01'] < 10.0) &
    (data_right_pm1pix_df['pc02'] > -5.0) &
    (data_right_pm1pix_df['pc02'] < 5.0) & 
    (data_right_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_right_green_at_red_cluster_pc01_5to10.csv")
print(outfile_full)
data_right_green_pc01_5to10_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_right_green_at_red_cluster_pc01_5to10"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_right_green_pc01_5to10_df.index:
    file_name = data_right_green_pc01_5to10_df.loc[irow, "file"]
    tz_x = data_right_green_pc01_5to10_df.loc[irow, "tz_x"]
    tz_y = data_right_green_pc01_5to10_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


# 10 -- 15
data_right_green_pc01_10to15_df = data_right_pm1pix_df[
    (data_right_pm1pix_df['pc01'] > 10.0) &
    (data_right_pm1pix_df['pc01'] < 15.0) &
    (data_right_pm1pix_df['pc02'] > -5.0) &
    (data_right_pm1pix_df['pc02'] < 5.0) & 
    (data_right_pm1pix_df["nstar_cat8"] == 0)]

outdir = indir
outfile_full = (outdir + "/"
                + "sel_pca_right_green_at_red_cluster_pc01_10to15.csv")
print(outfile_full)
data_right_green_pc01_10to15_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_right_green_at_red_cluster_pc01_10to15"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for irow in data_right_green_pc01_10to15_df.index:
    file_name = data_right_green_pc01_10to15_df.loc[irow, "file"]
    tz_x = data_right_green_pc01_10to15_df.loc[irow, "tz_x"]
    tz_y = data_right_green_pc01_10to15_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data)
    fig.colorbar(axi, ax=ax)
    title = f"{file_name}"
    xlabel = f"tz_x = {tz_x}"
    ylabel = f"tz_y = {tz_y}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    axi = ax.imshow(hdu0.data, norm="log")
    fig.colorbar(axi, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    outfile_full = (outdir + "/" + file_name + "_log.png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()


print(data_left_green_pc01_0to5_df["tz_x"].value_counts())
print(data_left_green_pc01_0to5_df["tz_y"].value_counts())
print(data_left_green_pc01_5to10_df["tz_x"].value_counts())
print(data_left_green_pc01_5to10_df["tz_y"].value_counts())
print(data_left_green_pc01_10to15_df["tz_x"].value_counts())
print(data_left_green_pc01_10to15_df["tz_y"].value_counts())

print(data_right_green_pc01_0to5_df["tz_x"].value_counts())
print(data_right_green_pc01_0to5_df["tz_y"].value_counts())
print(data_right_green_pc01_5to10_df["tz_x"].value_counts())
print(data_right_green_pc01_5to10_df["tz_y"].value_counts())
print(data_right_green_pc01_10to15_df["tz_x"].value_counts())
print(data_right_green_pc01_10to15_df["tz_y"].value_counts())

