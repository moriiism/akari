#
# add_dist_lr_pca.py
#
# calc distance between left and right image
# measured with pca vector.
#

import sys
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
import pickle as pk

# import seaborn as sns # visualize

from akarilib import calc_dist_lr_for_pca_in_row_of_dataframe

# python3 add_flag_pca.py 0 0 1 -5 40 -20 20

flag_cat = 0
pca_feature = 0
use_prefit = 0
args = sys.argv
nargs = len(args) - 1
print(nargs)
if (7 == nargs):
    flag_cat = int(args[1])
    pca_feature = int(args[2])
    use_prefit = int(args[3])
    pc01_lo_str = args[4]
    pc01_up_str = args[5]
    pc02_lo_str = args[6]
    pc02_up_str = args[7]
    print("flag_cat = ", flag_cat)
    print("pca_feature = ", pca_feature)
    print("use_prefit = ", use_prefit)
    print("pc01_lo_str = ", pc01_lo_str)
    print("pc01_up_str = ", pc01_up_str)
    print("pc02_lo_str = ", pc02_lo_str)
    print("pc02_up_str = ", pc02_up_str)
else:
    print('usage: python3 add_flag_pca.py flag_cat pca_feature use_prefit ' +
          'pc01_lo pc01_up pc02_lo pc02_up')
    print('usage: flag_cat means that csv files contain ' +
          'catalog(1) or not(0).')
    print('usage: pca_feature means type of features for pca calculation.')
    print('usage: use_prefit means that use prefit(1) or not(0).')
    print('usage: arg4, 5, 6, 7 means ' +
          'pc01_lo pc01_up pc02_lo pc02_up.' +
          'Set 4 values or def def def def.')
    print('Arguments are not 7.')
    exit()
    
# tag
pca_tag_str = ("pca_cat%d_ftr%d_prefit%d_tmp" %
               (flag_cat, pca_feature, use_prefit))

indir = os.environ["AKARI_ANA_DIR"] + "/" + pca_tag_str
incsv = ""
if (0 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star_pca.csv"
elif (1 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star_cat_pca.csv"
else:
    print("bad flag_cat = ", flag_cat)
    exit()


# read input
data_df = pd.read_csv(incsv)
print(data_df)

data_selrow_left_df = data_df[(data_df["left"] == 1) & 
                              (data_df["dark"] == 0) &
                              (data_df["edge"] == 0)]
data_selrow_right_df = data_df[(data_df["left"] == 0) & 
                               (data_df["dark"] == 0) &
                               (data_df["edge"] == 0)]

dist_df = data_selrow_left_df.apply(
    calc_dist_lr_for_pca_in_row_of_dataframe,
    right_df=data_selrow_right_df, axis=1)


data_dist_df = pd.concat([data_df, dist_df], axis=1)

# output
outdir = indir
outcsv = ""
if (0 == flag_cat):
    outcsv = outdir + "/" + "akari_stat_fit_star_pca_dist.csv"
elif (1 == flag_cat):
    outcsv = outdir + "/" + "akari_stat_fit_star_cat_pca_dist.csv"
else:
    print("bad flag_cat = ", flag_cat)
    exit()
    
print(f"outcsv = {outcsv}")
data_dist_df.to_csv(outcsv, index=False)


# plot

plt.scatter(data_dist_df["pc01"],
            data_dist_df["dist_lr_pca"],
            s=1, c="b")
plt.title("dist_lr_pca v.s. pc01")
plt.xlabel("pc01")
plt.ylabel("dist_lr_pca")
plt.grid(True, linestyle='--')

outfile_full = outdir + "/" + "dist_pc01.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

# plot
plt.scatter(data_dist_df["dist_lr_pca"],
            data_dist_df["nfind"],
            s=1, c="b")
plt.title("dist_lr_pca v.s. nfind")
plt.xlabel("dist_lr_pca")
plt.ylabel("nfind")
plt.grid(True, linestyle='--')

outfile_full = outdir + "/" + "dist_nfind.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


# plot
plt.scatter(data_dist_df["dist_lr_pca"],
            data_dist_df["nstar_cat8"],
            s=1, c="b")
plt.title("dist_lr_pca v.s. nstar_cat8")
plt.xlabel("dist_lr_pca")
plt.ylabel("nstar_cat8")
plt.grid(True, linestyle='--')

outfile_full = outdir + "/" + "dist_nstar_cat8.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

