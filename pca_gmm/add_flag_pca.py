#
# add_flag_pca.py
#
# add pca components
#

# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool/preproc/fits_to_csv.py
#   % python $akari_tool/preproc/add_stat.py
#   % python $akari_tool/preproc/add_stat_fit.py
#   % python $akari_tool/preproc/add_flag_star_pos.py
#   % python $akari_tool/preproc/add_flag_pca.py
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

from akarilib import get_colname_lst_of_pixarr_norm

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
pca_tag_str = ("pca_cat%d_ftr%d_prefit%d" %
               (flag_cat, pca_feature, use_prefit))

indir = os.environ["AKARI_ANA_DIR"]
incsv = ""
if (0 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star.csv"
elif (1 == flag_cat):
    incsv = indir + "/" + "akari_stat_fit_star_cat.csv"
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

data_selrow_df = data_df[(data_df["left"] == 1) & 
                         (data_df["dark"] == 0) &
                         (data_df["edge"] == 0) &
                         (data_df["star_pos"] > 1)]

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
colname_lst = None
if (0 == pca_feature):
    colname_lst = (
        colname_pixarr_norm_lst + ["sum",
                                   "norm_stddev",
                                   "norm_min",
                                   "norm_max",
                                   "norm_skew",
                                   "norm_kurt",
                                   "norm_gini",
                                   "ratio_around_to_peak",
                                   "gfit_mu_x",
                                   "gfit_mu_y",
                                   "gfit_sigma_x",
                                   "gfit_sigma_y",
                                   "gfit_theta",
                                   "gfit_norm",
                                   "gfit_const",
                                   "gfit_valid"])
elif (1 == pca_feature):
    colname_lst = (
        colname_pixarr_norm_lst + ["sum",
                                   "norm_stddev",
                                   "norm_min",
                                   "norm_max",
                                   "norm_skew",
                                   "norm_kurt",
                                   "norm_gini",
                                   "ratio_around_to_peak",
                                   "gfit_mu_x",
                                   "gfit_mu_y",
                                   "gfit_sigma_x",
                                   "gfit_sigma_y",
                                   "gfit_theta",
                                   "gfit_norm",
                                   "gfit_const",
                                   "gfit_valid",
                                   "norm_ave_margin"])
elif (2 == pca_feature):
    colname_lst = (
        colname_pixarr_norm_lst + ["sum",
                                   "norm_stddev",
                                   "norm_min",
                                   "norm_max",
                                   "norm_skew",
                                   "norm_kurt",
                                   "norm_gini",
                                   "ratio_around_to_peak"])
elif (3 == pca_feature):
    colname_lst = (
        colname_pixarr_norm_lst + ["sum",
                                   "norm_stddev",
                                   "norm_min",
                                   "norm_max",
                                   "norm_skew",
                                   "norm_kurt",
                                   "norm_gini",
                                   "ratio_around_to_peak",
                                   "norm_ave_margin"])
elif (4 == pca_feature):
    colname_lst = (
        colname_pixarr_norm_lst + ["sum",
                                   "norm_stddev",
                                   "norm_min",
                                   "norm_max",
                                   "norm_skew",
                                   "norm_kurt",
                                   "norm_gini",
                                   "ratio_around_to_peak",
                                   "gfit_mu_x",
                                   "gfit_mu_y",
                                   "gfit_sigma_x",
                                   "gfit_sigma_y",
                                   "gfit_theta",
                                   "gfit_norm",
                                   "gfit_const",
                                   "gfit_valid",
                                   "ave_margin"])
else:
    print("bad pca_feature %d" % pca_feature)
    exit()

# to make model of pca
data_selrow_selcol_df = data_selrow_df[colname_lst]
print(data_selrow_selcol_df.columns)
sc = StandardScaler()
data_selrow_selcol_sc_ndarr = sc.fit_transform(
    data_selrow_selcol_df)
print("data_selrow_selcol_sc_ndarr")
print(data_selrow_selcol_sc_ndarr.shape)
ncomp_pca = data_selrow_selcol_sc_ndarr.shape[1]
print("ncomp_pca = ", ncomp_pca)

# to apply pca model
data_selcol_df = data_df[colname_lst]
print(data_selcol_df.columns)
data_selcol_sc_ndarr = sc.fit_transform(
    data_selcol_df)

# pca
pca = None
if (use_prefit == 0):
    pca = PCA(n_components=ncomp_pca)
    pca.fit(data_selrow_selcol_sc_ndarr)
    out_pkl = outdir + "/" + "pca.pkl"
    pk.dump(pca, open(out_pkl,"wb"))
elif (use_prefit == 1):
    flag_cat_for_making_model = 1
    use_prefit_for_making_model = 0
    pca_prefit_tag_str = ("pca_cat%d_ftr%d_prefit%d" %
                          (flag_cat_for_making_model,
                           pca_feature,
                           use_prefit_for_making_model))
    prefit_pkl = (os.environ["MODEL_DIR"] + "/"
                  + pca_prefit_tag_str + "/"
                  + "pca.pkl")
    pca = pk.load(open(prefit_pkl,'rb'))
else:
    print("bad use_prefit = ", use_prefit)
    exit()

# apply pca
pca_ndarr = pca.transform(data_selcol_sc_ndarr)
print(pca_ndarr.shape)

colname_pca_lst = [f"pc{i+1:02}"  for i in range(ncomp_pca)]
print(colname_pca_lst)
data_pca_df = pd.concat([data_df,
                         pd.DataFrame(pca_ndarr[:,0:ncomp_pca],
                                      columns=colname_pca_lst,
                                      index=data_df.index)],
                        axis = 1)

# output
outcsv = ""
if (0 == flag_cat):
    outcsv = outdir + "/" + "akari_stat_fit_star_pca.csv"
elif (1 == flag_cat):
    outcsv = outdir + "/" + "akari_stat_fit_star_cat_pca.csv"
else:
    print("bad flag_cat = ", flag_cat)
    exit()
    
print(f"outcsv = {outcsv}")
data_pca_df.to_csv(outcsv, index=False)


# plot

### scatter plot range
pc01_lo = 0.0
pc01_up = 0.0
pc02_lo = 0.0
pc02_up = 0.0
if ((pc01_lo_str == "def") &
    (pc01_up_str == "def") &
    (pc02_lo_str == "def") &
    (pc02_up_str == "def")):
    pc01_min = data_pca_df['pc01'].min()
    pc01_max = data_pca_df['pc01'].max()
    pc02_min = data_pca_df['pc02'].min()
    pc02_max = data_pca_df['pc02'].max()
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

plt.scatter(data_pca_df['pc01'], data_pca_df['pc02'], s=1, c="b")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
plt.grid(True, linestyle='--')

# data for model
data_pca_selrow_df = data_pca_df[(data_pca_df["left"] == 1) & 
                                 (data_pca_df["dark"] == 0) &
                                 (data_pca_df["edge"] == 0) &
                                 (data_pca_df["star_pos"] > 1)]
plt.scatter(data_pca_selrow_df['pc01'],
            data_pca_selrow_df['pc02'], s=1, c="r")
plt.title("PC (Blue: all, Red: selected for making PCA model)")

outfile_full = outdir + "/" + "pca.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

# pca contribution plot

pca_contrib = pd.DataFrame(pca.explained_variance_ratio_,
                           index=["PC{}".format(x + 1)
                                  for x in range(ncomp_pca)])
print(pca_contrib)
plt.plot([i for i in range(ncomp_pca + 1)],
         [0] + list(np.cumsum(pca.explained_variance_ratio_)),
         marker="o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid(True, linestyle='--')

outfile_full = outdir + "/" + "pca_contrib.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


# pca components
pca_comp_df = pd.DataFrame(pca.components_, 
                           columns=data_selcol_df.columns,
                           index=["PC{}".format(x + 1)
                                  for x in range(ncomp_pca)])
print(pca_comp_df)
for x, y, name in zip(pca.components_[0], pca.components_[1],
                      data_selcol_df.columns):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid(True, linestyle='--')
plt.xlabel("PC1")
plt.ylabel("PC2")

# sort by pc0 
# sort by pc1 
pc01_name_2darr = np.stack([pca.components_[0],
                            pca.components_[1],
                            data_selcol_df.columns], 1)
pc01_name_sort_by_pc0_2darr = pc01_name_2darr[
    np.argsort(pc01_name_2darr[:,0])]
pc01_name_sort_by_pc1_2darr = pc01_name_2darr[
    np.argsort(pc01_name_2darr[:,1])]
outfile_full = outdir + "/" + "pca_comp_sort_by_pc0.txt"
np.savetxt(outfile_full, pc01_name_sort_by_pc0_2darr, 
           fmt=['%.5e', '%.5e', '%s'])
outfile_full = outdir + "/" + "pca_comp_sort_by_pc1.txt"
np.savetxt(outfile_full, pc01_name_sort_by_pc1_2darr,
           fmt=['%.5e', '%.5e', '%s'])

outfile_full = outdir + "/" + "pca_comp.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()
