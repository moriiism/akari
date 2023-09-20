#
# add_flag_pca.py
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
incsv = indir + "/" + "akari_stat_fit_star_cat.csv"
data_df = pd.read_csv(incsv)
print(data_df)

# remove dark and edge
print(len(data_df))
data_selrow_df = data_df[(data_df["dark"]==0) &
                         (data_df["tz_y"] > 5)]
# data_selrow_df = data_df[(data_df["dark"]==0)]


print(len(data_selrow_df))

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
colname_lst = colname_pixarr_norm_lst + ["sum",
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
                                         "gfit_valid"]

data_selrow_selcol_df = data_selrow_df[colname_lst]
print(data_selrow_selcol_df.columns)
data_selcol_df = data_df[colname_lst]
print(data_selcol_df.columns)

sc = StandardScaler()
data_selrow_selcol_sc_ndarr = sc.fit_transform(
    data_selrow_selcol_df)
data_selcol_sc_ndarr = sc.fit_transform(
    data_selcol_df)

print("data_selrow_selcol_sc_ndarr")
print(data_selrow_selcol_sc_ndarr.shape)

ncomp_pca = data_selrow_selcol_sc_ndarr.shape[1]
print("ncomp_pca = ", ncomp_pca)

# pca
pca = PCA(n_components=ncomp_pca)
# pca_ndarr = pca.fit_transform(data_selrow_selcol_sc_ndarr)
pca.fit(data_selrow_selcol_sc_ndarr)
pca_ndarr = pca.transform(data_selcol_sc_ndarr)
print(pca_ndarr.shape)

# pca plot
colname_pca_lst = [f"pc{i+1:02}"  for i in range(ncomp_pca)]
print(colname_pca_lst)
data_pca_df = pd.concat([data_df,
                         pd.DataFrame(pca_ndarr[:,0:ncomp_pca],
                                      columns=colname_pca_lst,
                                      index=data_df.index)],
                        axis = 1)
plt.scatter(data_pca_df['pc01'], data_pca_df['pc02'], s=1, c="b")
plt.title("PC")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.grid(True, linestyle='--')

outdir = indir
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

outdir = indir
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


outdir = indir
outcsv = outdir + "/" + "akari_stat_fit_star_cat_pca.csv"
print(f"outcsv = {outcsv}")
data_pca_df.to_csv(outcsv, index=False)


