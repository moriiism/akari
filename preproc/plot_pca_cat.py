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

data_cat_df = data_df[data_df["nstar_cat10"] > 0]
data_nocat_df = data_df[data_df["nstar_cat10"] == 0]

plt.scatter(data_nocat_df['pc01'], data_nocat_df['pc02'], s=1, c="b")
plt.scatter(data_cat_df['pc01'], data_cat_df['pc02'], s=1, c="r")
plt.title("PC")
plt.xlabel("pc01")
plt.ylabel("pc02")
plt.grid(True, linestyle='--')

outdir = indir
outfile_full = outdir + "/" + "pca_cat.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()
