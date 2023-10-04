#
# check_tzl_y.py
#
# When pc02 vs pc01 is plotted,
# there is a cluster by the edge.
# The edge is characterized by tzl_y value,
# So, the histogram of tzl_y value is made.
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
#   % python $akari_tool/plot_pca_cat/check_tzl_y.py
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
                       (data_df["dark"] == 0)]
data_right_df = data_df[(data_df["left"] == 0) &
                        (data_df["dark"] == 0)]

#### cluster pc01>4 & pc02<5

data_left_cluster_df = data_left_df[(data_left_df["pc01"] < -5) & 
                                    (data_left_df["pc02"] > 0) &
                                    (data_left_df["pc02"] < 10)]
data_right_cluster_df = data_right_df[(data_right_df["pc01"] < -5) &
                                      (data_right_df["pc02"] > 0) &
                                      (data_right_df["pc02"] < 10)]

print(data_left_cluster_df[["tzl_y"]].value_counts())
print(data_right_cluster_df[["tzl_y"]].value_counts())

print(data_left_cluster_df[["tz_y"]].value_counts())
print(data_right_cluster_df[["tz_y"]].value_counts())

exit()



data_left_cat_df = data_left_df[(data_left_df["nstar_cat8"] > 0)]
data_left_pm1pix_df = data_left_df[(data_left_df["star_pos"] > 1)]
data_left_pm2pix_df = data_left_df[(data_left_df["star_pos"] > 0)]

data_right_cat_df = data_right_df[(data_right_df["nstar_cat8"] > 0)]
data_right_pm1pix_df = data_right_df[(data_right_df["star_pos"] > 1)]
data_right_pm2pix_df = data_right_df[(data_right_df["star_pos"] > 0)]

### plot left
plt.scatter(data_left_df['pc01'], data_left_df['pc02'],
            s=1, c="black")
plt.scatter(data_left_pm2pix_df['pc01'], data_left_pm2pix_df['pc02'],
            s=1, c="blue")
plt.scatter(data_left_pm1pix_df['pc01'], data_left_pm1pix_df['pc02'],
            s=1, c="green")
plt.scatter(data_left_cat_df['pc01'], data_left_cat_df['pc02'],
            s=1, c="red")
plt.title("PC")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "pca_left_cat.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

### plot right
plt.scatter(data_right_df['pc01'], data_right_df['pc02'],
            s=1, c="black")
plt.scatter(data_right_pm2pix_df['pc01'], data_right_pm2pix_df['pc02'],
            s=1, c="blue")
plt.scatter(data_right_pm1pix_df['pc01'], data_right_pm1pix_df['pc02'],
            s=1, c="green")
plt.scatter(data_right_cat_df['pc01'], data_right_cat_df['pc02'],
            s=1, c="red")
plt.title("PC")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "pca_right_cat.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()



