#
# plot_pca.py
#
# for analysis shown in the report 2024.04.04
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

# usage:
#  python3 plot_pca.py 1 0 0 def def def def

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
    print('usage: python3 plot_pca.py flag_cat pca_feature use_prefit ' +
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
pca_tag_str = ("pca_cat%d_ftr%d_prefit%d_20240404" %
               (flag_cat, pca_feature, use_prefit))

indir = os.environ["AKARI_ANA_DIR"]
incsv = ""
if (1 == flag_cat):
    incsv = (indir + "/" + pca_tag_str + "/"
             + "akari_stat_fit_star_cat_pca.csv")
else:
    print("bad flag_cat = ", flag_cat)
    exit()

# for output
outdir = indir + "/" + pca_tag_str
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

# read input
data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["left"] == 1) & 
                       (data_df["dark"] == 0) &
                       (data_df["edge"] == 0)]
data_left_df.reset_index(inplace=True, drop=True)
data_right_df = data_df[(data_df["left"] == 0) & 
                        (data_df["dark"] == 0) &
                        (data_df["edge"] == 0)]
data_right_df.reset_index(inplace=True, drop=True)

# (data_df["star_pos"] > 1)]                        

data_2darr = data_left_df[["pc01", "pc02"]].values
print(data_2darr)
print(type(data_2darr))


###  scatter plot range
pc01_lo = 0.0
pc01_up = 0.0
pc02_lo = 0.0
pc02_up = 0.0
if ((pc01_lo_str == "def") &
    (pc01_up_str == "def") &
    (pc02_lo_str == "def") &
    (pc02_up_str == "def")):
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
else:
    pc01_lo = float(pc01_lo_str)
    pc01_up = float(pc01_up_str)
    pc02_lo = float(pc02_lo_str)
    pc02_up = float(pc02_up_str)

print(pc01_lo, pc01_up, pc02_lo, pc02_up)


# by star_pos (left-right match): lrm
data_lrm_spike_df = data_left_df[data_left_df["star_pos"]<=1]
plt.scatter(data_lrm_spike_df['pc01'],
            data_lrm_spike_df['pc02'],
            s=1, c="b")
data_lrm_star_df = data_left_df[data_left_df["star_pos"]>1]
plt.scatter(data_lrm_star_df['pc01'],
            data_lrm_star_df['pc02'],
            s=1, c="r")

plt.title("Red: star: star_pos>1, Blue: spike: star_pos<=1")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
plt.grid(True, linestyle='--')

# plot
outfile_full = outdir + "/" + "pca_star_pos.png"
print("outfile = ", outfile_full)

plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)

plt.cla()
plt.clf()


# plot by catalog
data_cat_spike_df = data_left_df[data_left_df["nstar_cat8"]==0]
plt.scatter(data_cat_spike_df['pc01'],
            data_cat_spike_df['pc02'],
            s=1, c="b")
data_cat_star_df = data_left_df[data_left_df["nstar_cat8"]>0]
plt.scatter(data_cat_star_df['pc01'],
            data_cat_star_df['pc02'],
            s=1, c="r")

plt.title("Red: nstar_cat8 > 0, Blue: nstar_cat8 = 0")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
plt.grid(True, linestyle='--')

# plot
outfile_full = outdir + "/" + "pca_cat.png"
print("outfile = ", outfile_full)

plt.savefig(outfile_full,
            bbox_inches='tight',  pad_inches=0.1)

#######

data_cat0_lrm0_df = None
data_cat0_lrm1_df = None
data_cat1_lrm0_df = None
data_cat1_lrm1_df = None
data_cat0_df = None
data_cat1_df = None
data_lrm0_df = None
data_lrm1_df = None
nevt_true_negative = 0
nevt_false_positive = 0
nevt_false_negative = 0
nevt_true_positive = 0
nevt_cat0 = 0
nevt_cat1 = 0
nevt_lrm0 = 0
nevt_lrm1 = 0

# cat 0/1: "nstar_cat8"==0 / "nstar_cat8">0
# lrm 0/1: "star_pos"<=1 / "star_pos">1

data_cat0_lrm0_df = data_left_df[(data_left_df["nstar_cat8"]==0) &
                                 (data_left_df["star_pos"]<=1)]
data_cat0_lrm1_df = data_left_df[(data_left_df["nstar_cat8"]==0) &
                                 (data_left_df["star_pos"]>1)]
data_cat1_lrm0_df = data_left_df[(data_left_df["nstar_cat8"]>0) &
                                 (data_left_df["star_pos"]<=1)]
data_cat1_lrm1_df = data_left_df[(data_left_df["nstar_cat8"]>0) &
                                 (data_left_df["star_pos"]>1)]
data_cat0_df = data_left_df[(data_left_df["nstar_cat8"]==0)]
data_cat1_df = data_left_df[(data_left_df["nstar_cat8"]>0)]
data_lrm0_df = data_left_df[(data_left_df["star_pos"]<=1)]
data_lrm1_df = data_left_df[(data_left_df["star_pos"]>1)]

nevt_true_negative = len(data_cat0_lrm0_df)
nevt_false_positive = len(data_cat0_lrm1_df)
nevt_false_negative = len(data_cat1_lrm0_df)
nevt_true_positive = len(data_cat1_lrm1_df)
nevt_cat0 = len(data_cat0_df)
nevt_cat1 = len(data_cat1_df)
nevt_lrm0 = len(data_lrm0_df)
nevt_lrm1 = len(data_lrm1_df)
nevt_all  = len(data_left_df)

print("cat0, lrm0 (true negative)  = ", nevt_true_negative)
print("cat0, 1rm1 (false positive) = ", nevt_false_positive)
print("cat1, lrm0 (false negative) = ", nevt_false_negative)
print("cat1, lrm1 (true positive)  = ", nevt_true_positive)

true_negative_rate = nevt_true_negative / nevt_cat0
false_positive_rate = nevt_false_positive / nevt_cat0
false_negative_rate = nevt_false_negative / nevt_cat1
true_positive_rate = nevt_true_positive / nevt_cat1

print("true_negative_rate = ", true_negative_rate)
print("false_positive_rate = ", false_positive_rate)
print("false_negative_rate = ", false_negative_rate)
print("true_positive_rate = ", true_positive_rate)

print("confusion matrix:")
print("               spike    star    total")
print(f"lrm_spike: {nevt_true_negative} {nevt_false_negative} {nevt_lrm0} ")
print(f"lrm_star:  {nevt_false_positive} {nevt_true_positive} {nevt_lrm1} ")
print(f"total:     {nevt_cat0} {nevt_cat1}  {nevt_all}")
    
print(len(data_cat0_lrm0_df) + len(data_cat0_lrm1_df)
      + len(data_cat1_lrm0_df) + len(data_cat1_lrm1_df))
print(len(data_left_df))

