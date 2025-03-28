#
# plot_pca_cat.py
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
#   % python $akari_tool/preproc/plot_pca_cat.py
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
                       (data_df["dark"] == 0) &
                       (data_df["tz_y"] > 5)]
data_right_df = data_df[(data_df["left"] == 0) &
                        (data_df["dark"] == 0) &
                        (data_df["tz_y"] > 5)]

#data_left_df = data_df[(data_df["left"] == 1) &
#                       (data_df["dark"] == 0)]
#data_right_df = data_df[(data_df["left"] == 0) &
#                        (data_df["dark"] == 0)]


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

print(pc01_lo, pc01_up, pc02_lo, pc02_up)

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
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
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
plt.xlim(pc01_lo, pc01_up)
plt.ylim(pc02_lo, pc02_up)
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "pca_right_cat.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()

