#
# gmm.py
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
#   % python $akari_tool/plog_gmm/gmm.py

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

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit_star_cat_pca.csv"
data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["tz_x"] < 64) & 
                       (data_df["dark"] == 0)]
data_left_df.reset_index(inplace=True, drop=True)

data_2darr = data_left_df[["pc01", "pc02", "pc03"]].values
print(data_2darr)
print(type(data_2darr))

# gmm

# 
# random_state=0,
#                              
#      init_params='kmeans',
#                              means_init=[[-3.0, -1.0],
#                                          [+0.0, +5.0],
#                                          [+5.5, -1.0],
#                                          [+6.5, -1.0],
#                                          [-6.0, +4.0]],

gmm = mixture.GaussianMixture(n_components=5,
                              covariance_type="full",
                              verbose=5)
cluster_gmm = gmm.fit_predict(data_2darr)
cluster_prob = gmm.predict_proba(data_2darr)

data_gmm_df = pd.concat([data_left_df,
                         pd.DataFrame(cluster_gmm,
                                      columns=['cluster_gmm']),
                         pd.DataFrame(cluster_prob,
                                      columns=['cluster_prob_01',
                                               'cluster_prob_02',
                                               'cluster_prob_03',
                                               'cluster_prob_04',
                                               'cluster_prob_05'])],
                        axis=1)

print(data_gmm_df[['pc01', 'pc02', "pc03", "cluster_gmm"]])


data_gmm_cl0_df = data_gmm_df[data_gmm_df["cluster_gmm"]==0]
plt.scatter(data_gmm_cl0_df['pc01'],
            data_gmm_cl0_df['pc02'],
            s=1, c="b")
data_gmm_cl1_df = data_gmm_df[data_gmm_df["cluster_gmm"]==1]
plt.scatter(data_gmm_cl1_df['pc01'],
            data_gmm_cl1_df['pc02'],
            s=1, c="r")
data_gmm_cl2_df = data_gmm_df[data_gmm_df["cluster_gmm"]==2]
plt.scatter(data_gmm_cl2_df['pc01'],
            data_gmm_cl2_df['pc02'],
            s=1, c="g")
data_gmm_cl3_df = data_gmm_df[data_gmm_df["cluster_gmm"]==3]
plt.scatter(data_gmm_cl3_df['pc01'],
            data_gmm_cl3_df['pc02'],
            s=1, c="black")
data_gmm_cl4_df = data_gmm_df[data_gmm_df["cluster_gmm"]==4]
plt.scatter(data_gmm_cl4_df['pc01'],
            data_gmm_cl4_df['pc02'],
            s=1, c="yellow")

plt.title("Gaussian Mixture")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "gmm_3d_5cl.png"
print("outfile = ", outfile_full)

plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)


#data_star0_gmm0_df = data_gmm_df[
#    (data_gmm_df["star"]==0) & (data_gmm_df["cluster_gmm"]==0) ]
#data_star0_gmm1_df = data_gmm_df[
#    (data_gmm_df["star"]==0) & (data_gmm_df["cluster_gmm"]==1) ]
#data_star1_gmm0_df = data_gmm_df[
#    (data_gmm_df["star"]==1) & (data_gmm_df["cluster_gmm"]==0) ]
#data_star1_gmm1_df = data_gmm_df[
#    (data_gmm_df["star"]==1) & (data_gmm_df["cluster_gmm"]==1) ]
#
#
#print("star0, gmm0 (true negative)  = ", len(data_star0_gmm0_df))
#print("star0, gmm1 (false positive) = ", len(data_star0_gmm1_df))
#print("star1, gmm0 (false negative) = ", len(data_star1_gmm0_df))
#print("star1, gmm1 (true positive)  = ", len(data_star1_gmm1_df))
#
#print(len(data_star0_gmm0_df) + len(data_star0_gmm1_df)
#      + len(data_star1_gmm0_df) + len(data_star1_gmm1_df))
#print(len(data_gmm_df))




