#
# gmm_margin.py
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
#   % python $akari_tool/preproc/add_flag_pca_margin.py
#   % python $akari_tool/preproc/add_flag_star_catalog.py
#   % python $akari_tool/plot_gmm/gmm_margin_2d_2cl.py

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

# python gmm_2d_2cl.py 0 1 -5 40 -20 20



flag_cat = 0
use_prefit = 0
args = sys.argv
nargs = len(args) - 1
print(nargs)
if (6 == nargs):
    flag_cat = int(args[1])
    use_prefit = int(args[2])
    pc01_lo_str = args[3]
    pc01_up_str = args[4]
    pc02_lo_str = args[5]
    pc02_up_str = args[6]
    print("flag_cat = ", flag_cat)
    print("use_prefit = ", use_prefit)
    print("pc01_lo_str = ", pc01_lo_str)
    print("pc01_up_str = ", pc01_up_str)
    print("pc02_lo_str = ", pc02_lo_str)
    print("pc02_up_str = ", pc02_up_str)
    pass
else:
    print('usage: python gmm_margin_2d_2cl.py 1/0 1/0 ' +
          'pc01_lo pc01_up pc02_lo pc02_up')
    print('usage: arg1 means that csv files contain ' +
          'catalog(1) or not(0).')
    print('usage: arg2 means that use prefit(1) or not(0).')
    print('usage: arg3, 4, 5, 6 means ' +
          'pc01_lo pc01_up pc02_lo pc02_up.' +
          'Set 4 values or def def def def.')
    print('Arguments are not 6.')
    exit()

indir = os.environ["AKARI_ANA_DIR"]
incsv = ""
if (0 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star_pca_margin.csv"
elif (1 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star_pca_margin_cat.csv"
else:
    print("bad flag_cat = ", flag_cat)
    exit()

data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["left"] == 1) & 
                       (data_df["dark"] == 0) &
                       (data_df["tz_y"] > 5) &
                       (data_df["tzl_x"] > 7) &
                       (data_df["star_pos"] > 1)]
data_left_df.reset_index(inplace=True, drop=True)
data_right_df = data_df[(data_df["left"] == 0) & 
                        (data_df["dark"] == 0) &
                        (data_df["tz_y"] > 5) &
                        (data_df["tzl_x"] > 7) &
                        (data_df["star_pos"] > 1)]
data_right_df.reset_index(inplace=True, drop=True)

data_2darr = data_left_df[["pc01", "pc02"]].values
print(data_2darr)
print(type(data_2darr))


### get min and max for scatter plot range

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

    print(pc01_lo, pc01_up, pc02_lo, pc02_up)

else:
    pc01_lo = float(pc01_lo_str)
    pc01_up = float(pc01_up_str)
    pc02_lo = float(pc02_lo_str)
    pc02_up = float(pc02_up_str)

    print(pc01_lo, pc01_up, pc02_lo, pc02_up)

# gmm

# 
# 
#                              
#      init_params='kmeans',
#                              means_init=[[-3.0, -1.0],
#                                          [+0.0, +5.0],
#                                          [+5.5, -1.0],
#                                          [+6.5, -1.0],
#                                          [-6.0, +4.0]],

gmm = None
if (use_prefit == 1):
    prefit_pkl = os.environ["GMM_MARGIN_MODEL"]
    gmm = pk.load(open(prefit_pkl, 'rb'))
elif (use_prefit == 0):
    gmm = mixture.GaussianMixture(n_components=2,
                                  random_state=1,
                                  covariance_type="full",
                                  verbose=5)
    gmm.fit(data_2darr)

else:
    print("bad use_prefit = ", use_prefit)
    exit()

cluster_gmm = gmm.predict(data_2darr)
cluster_prob = gmm.predict_proba(data_2darr)

data_gmm_df = pd.concat([data_left_df,
                         pd.DataFrame(cluster_gmm,
                                      columns=['cluster_gmm']),
                         pd.DataFrame(cluster_prob,
                                      columns=['cluster_prob_01',
                                               'cluster_prob_02'])],
                        axis=1)

print(data_gmm_df[['pc01', 'pc02', "cluster_gmm"]])


data_gmm_cl0_df = data_gmm_df[data_gmm_df["cluster_gmm"]==0]
plt.scatter(data_gmm_cl0_df['pc01'],
            data_gmm_cl0_df['pc02'],
            s=1, c="b")
data_gmm_cl1_df = data_gmm_df[data_gmm_df["cluster_gmm"]==1]
plt.scatter(data_gmm_cl1_df['pc01'],
            data_gmm_cl1_df['pc02'],
            s=1, c="r")

plt.title("Gaussian Mixture")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "gmm_margin_2d_2cl.png"
print("outfile = ", outfile_full)

plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)


data_gmm0_df = data_gmm_df[(data_gmm_df["cluster_gmm"] == 0)]
data_gmm1_df = data_gmm_df[(data_gmm_df["cluster_gmm"] == 1)]
nevt_gmm0 = len(data_gmm0_df)
nevt_gmm1 = len(data_gmm1_df)
nevt_all = len(data_gmm_df)
print("nevt_gmm0 = ", nevt_gmm0)
print("nevt_gmm1 = ", nevt_gmm1)
print("nevt_all = ", nevt_all)

data_star0_gmm0_df = None
data_star0_gmm1_df = None
data_star1_gmm0_df = None
data_star1_gmm1_df = None
data_star0_df = None
data_star1_df = None
nevt_true_negative = 0
nevt_false_positive = 0
nevt_false_negative = 0
nevt_true_positive = 0
nevt_star0 = 0
nevt_star1 = 0

if (1 == flag_cat):
    data_star0_gmm0_df = data_gmm_df[(data_gmm_df["nstar_cat8"] == 0) &
                                     (data_gmm_df["cluster_gmm"]==0)]
    data_star0_gmm1_df = data_gmm_df[(data_gmm_df["nstar_cat8"]==0) &
                                     (data_gmm_df["cluster_gmm"]==1) ]
    data_star1_gmm0_df = data_gmm_df[(data_gmm_df["nstar_cat8"] > 0) &
                                     (data_gmm_df["cluster_gmm"]==0) ]
    data_star1_gmm1_df = data_gmm_df[(data_gmm_df["nstar_cat8"] > 0) &
                                     (data_gmm_df["cluster_gmm"]==1) ]
    data_star0_df = data_gmm_df[(data_gmm_df["nstar_cat8"] == 0)]
    data_star1_df = data_gmm_df[(data_gmm_df["nstar_cat8"] > 0)]
    nevt_true_negative = len(data_star0_gmm0_df)
    nevt_false_positive = len(data_star0_gmm1_df)
    nevt_false_negative = len(data_star1_gmm0_df)
    nevt_true_positive = len(data_star1_gmm1_df)
    nevt_star0 = len(data_star0_df)
    nevt_star1 = len(data_star1_df)


    print("star0, gmm0 (true negative)  = ", nevt_true_negative)
    print("star0, gmm1 (false positive) = ", nevt_false_positive)
    print("star1, gmm0 (false negative) = ", nevt_false_negative)
    print("star1, gmm1 (true positive)  = ", nevt_true_positive)

    true_negative_rate = nevt_true_negative / nevt_star0
    false_positive_rate = nevt_false_positive / nevt_star0
    false_negative_rate = nevt_false_negative / nevt_star1
    true_positive_rate = nevt_true_positive / nevt_star1

    print("true_negative_rate = ", true_negative_rate)
    print("false_positive_rate = ", false_positive_rate)
    print("false_negative_rate = ", false_negative_rate)
    print("true_positive_rate = ", true_positive_rate)

    print("confusion matrix:")
    print("               spike    star    total")
    print(f"gmm_spike: {nevt_true_negative} {nevt_false_negative} {nevt_gmm0} ")
    print(f"gmm_star:  {nevt_false_positive} {nevt_true_positive} {nevt_gmm1} ")
    print(f"total:     {nevt_star0} {nevt_star1}  {nevt_all}")
    
    print(len(data_star0_gmm0_df) + len(data_star0_gmm1_df)
          + len(data_star1_gmm0_df) + len(data_star1_gmm1_df))
    print(len(data_gmm_df))

elif (0 == flag_cat):
    outdir = indir
    outcsv = outdir + "/" + "akari_stat_fit_star_pca_margin_gmm_2d_2cl.csv"
    print(f"outcsv = {outcsv}")
    data_gmm_df.to_csv(outcsv, index=False)

    outdir = indir
    outcsv = (outdir + "/"
              + "akari_stat_fit_star_pca_margin_gmm_2d_2cl_simple.csv")
    print(f"outcsv = {outcsv}")
    data_gmm_df[["file", "cluster_gmm",
                 "cluster_prob_01", "cluster_prob_02"]].to_csv(
                     outcsv, index=False)

else:
    print("bad flag_cat = ", flag_cat)
    exit()


if (use_prefit == 1):
    pass
elif (use_prefit == 0):
    out_pkl = indir + "/" + "gmm_margin.pkl"
    pk.dump(gmm, open(out_pkl,"wb"))
else:
    print("bad use_prefit = ", use_prefit)
    exit()


