#
# sel_pca_cluster.py
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
#   % python $akari_tool/plot_pca_cat/sel_pca_cluster/
#            sel_pca_cluster.py

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

data_right_cat_df = data_right_df[(data_right_df["nstar_cat8"] > 0)]
data_right_pm1pix_df = data_right_df[(data_right_df["star_pos"] > 1)]
data_right_pm2pix_df = data_right_df[(data_right_df["star_pos"] > 0)]

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


### select cluster

data_cluster_left_df = data_left_df[(data_left_df['pc01'] < 2.0) &
                                    (data_left_df['pc02'] > 4.0)]

outdir = indir
outfile_full = outdir + "/" + "sel_pca_cluster.csv"
print(outfile_full)
data_cluster_left_df.to_csv(outfile_full, index=False)

# plot images in a pca cluster 
outdir = outdir + "/" + "sel_pca_cluster"
if not os.path.exists(outdir):
    os.mkdir(outdir)

print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 8]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 9]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 20]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 22]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 28]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 33]
      [["tz_x", "file"]])
print(data_cluster_left_df[data_cluster_left_df["tz_x"] == 53]
      [["tz_x", "file"]])
exit()


for irow in data_cluster_left_df.index:
    file_name = data_cluster_left_df.loc[irow, "file"]
    tz_x = data_cluster_left_df.loc[irow, "tz_x"]
    tz_y = data_cluster_left_df.loc[irow, "tz_y"]
    print(file_name, tz_x, tz_y)
    data_dir = os.environ["AKARI_DATA_DIR"]
    file_name_full = data_dir + "/" + file_name
    print(file_name_full)

    fig, ax = plt.subplots(1, 1)
    hdu = fits.open(file_name_full)
    hdu0 = hdu[0]
    ax.imshow(hdu0.data)
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


print(data_cluster_left_df["tz_x"].value_counts())


