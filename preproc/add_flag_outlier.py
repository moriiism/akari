#
# pcasel.py
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib

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

from util import getColNameLst

indir = "/home/morii/work/akari/ana/spikethumb_20230407"
incsv = indir + "/" + "akari_stat_flag.csv"
data_df = pd.read_csv(incsv)
print(data_df)

# select 1000 events
#data_df = data_df.loc[0:1000,:]

# star_df = data_df[["star"]]

colname_drop_lst = getColNameLst()
colname_drop_lst.append("file")
colname_drop_lst.append("tzl_x")
colname_drop_lst.append("tzl_y")
colname_drop_lst.append("star")
data_sel_df = data_df.drop(colname_drop_lst, axis=1)
print(data_sel_df.columns)

sc = StandardScaler()
data_sc_ndarr = sc.fit_transform(data_sel_df)

print("data_sc_ndarr")
print(type(data_sc_ndarr))
print(data_sc_ndarr.shape)

ncomp_pca = data_sc_ndarr.shape[1]
print("ncomp_pca = ", ncomp_pca)

# pca
pca = PCA(n_components=ncomp_pca)
pca_ndarr = pca.fit_transform(data_sc_ndarr)

print(pca_ndarr)
print(type(pca_ndarr))

# contribution
pca_contrib = pd.DataFrame(pca.explained_variance_ratio_,
                           index=["PC{}".format(x + 1)
                                  for x in range(ncomp_pca)])
print(pca_contrib)

# plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")

plt.plot([i for i in range(ncomp_pca + 1)],
         [0] + list(np.cumsum(pca.explained_variance_ratio_)), marker="o")
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
pca_comp_df = pd.DataFrame(pca.components_, columns=data_sel_df.columns,
                           index=["PC{}".format(x + 1)
                                  for x in range(ncomp_pca)])
print(pca_comp_df)

# contribution plot
for x, y, name in zip(pca.components_[0], pca.components_[1],
                      data_sel_df.columns):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid(True, linestyle='--')
plt.xlabel("PC1")
plt.ylabel("PC2")

outfile_full = outdir + "/" + "pca_comp.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

# contribution plot zoom
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.xlim(0.2, 0.3)
plt.ylim(-0.05, 0.05)
for x, y, name in zip(pca.components_[0], pca.components_[1],
                      data_sel_df.columns):
    if ((0.2 < x) & (x < 0.3) &
        (-0.05 < y) & (y < 0.05)):
        plt.text(x, y, name)
plt.grid(True, linestyle='--')
plt.xlabel("PC1")
plt.ylabel("PC2")

outfile_full = outdir + "/" + "pca_comp_zoom.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()


# pca plot
colname_pca_lst = [f"pca{i+1:02}"  for i in range(ncomp_pca)]
print(colname_pca_lst)
data_pca_df = pd.concat([data_df,
                         pd.DataFrame(pca_ndarr[:,0:ncomp_pca],
                                      columns=colname_pca_lst)],
                        axis = 1)
plt.scatter(data_pca_df['pca01'], data_pca_df['pca02'], s=1, c="b")
plt.title("PCA")
plt.xlabel("pca01")
plt.ylabel("pca02")
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "pca.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

# pca zoom
plt.scatter(data_pca_df['pca01'], data_pca_df['pca02'], s=1, c="b")
plt.xlim(-10.0, 15.0)
plt.ylim(-10.0, 15.0)
outfile_full = outdir + "/" + "pca_zoom.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

print("len of data_pca_df = ", len(data_pca_df))

# remove outlier
data_pca_sel_df = data_pca_df[data_pca_df["pca01"] < 15]
print("len of data_pca_df = ", len(data_pca_sel_df))

outdir = "/home/morii/work/akari/ana/spikethumb_20230407"
outcsv = outdir + "/" + "akari_stat_flag_pcasel.csv"
print(f"outcsv = {outcsv}")
data_pca_sel_df.to_csv(outcsv, index=False)


# outlier id
data_pca_del_df = data_pca_df[data_pca_df["pca01"] >= 15]
print(data_pca_del_df)

outdir = "/home/morii/work/akari/ana/spikethumb_20230407"
outcsv = outdir + "/" + "akari_stat_flag_pcadel.csv"
print(f"outcsv = {outcsv}")
data_pca_del_df.to_csv(outcsv, index=False)

