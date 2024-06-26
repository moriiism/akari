#
# rf.py
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

# python3 rf.py

#args = sys.argv
#nargs = len(args) - 1
#print(nargs)
#if (0 == nargs):
#    exit()


indir = os.environ["AKARI_ANA_DIR"]
incsv = (indir + "/" + "akari_stat_fit_star_cat.csv")

# for output
outdir = indir + "/" + "rf"
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

# read input
data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["left"] == 1) & 
                       (data_df["dark"] == 0) &
                       (data_df["edge"] == 0) &
                       (data_df["star_pos"] > 1)]
data_left_df.reset_index(inplace=True, drop=True)
data_right_df = data_df[(data_df["left"] == 0) & 
                        (data_df["dark"] == 0) &
                        (data_df["edge"] == 0) &
                        (data_df["star_pos"] > 1)]
data_right_df.reset_index(inplace=True, drop=True)

print(data_left_df.columns)

