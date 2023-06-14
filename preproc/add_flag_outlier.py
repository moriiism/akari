#
# add_flag_outlier.py
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

from akarilib import get_colname_lst_of_pixarr

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_star_cat.csv"
data_df = pd.read_csv(incsv)
print(data_df)


#### !!!! drop is nor good operation !!!


colname_drop_lst = get_colname_lst_of_pixarr()
colname_drop_lst.append("file")
colname_drop_lst.append("tzl_x")
colname_drop_lst.append("tzl_y")
colname_drop_lst.append("crval1")
colname_drop_lst.append("crval2")
colname_drop_lst.append("ra")
colname_drop_lst.append("dec")
colname_drop_lst.append("star_pos")
colname_drop_lst.append("star_cat")

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

# pca plot
colname_pca_lst = [f"pc{i+1:02}"  for i in range(ncomp_pca)]
print(colname_pca_lst)
data_pca_df = pd.concat([data_df,
                         pd.DataFrame(pca_ndarr[:,0:ncomp_pca],
                                      columns=colname_pca_lst)],
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

# pca zoom
plt.scatter(data_pca_df['pc01'], data_pca_df['pc02'], s=1, c="b")
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

outlier_ser = (data_pca_df["pc01"] >= 15).astype("int")
outlier_ser.name = "outlier"
data_pca_outlier_df = pd.concat([data_pca_df, outlier_ser], axis=1)

outdir = indir
outcsv = outdir + "/" + "akari_stat_star_cat_outlier.csv"
print(f"outcsv = {outcsv}")
data_pca_outlier_df.to_csv(outcsv, index=False)
