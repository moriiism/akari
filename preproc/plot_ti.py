#
# plot_ti.py
#
# plot ti vs file id
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
#   % python $akari_tool/preproc/plot_ti.py
#

import os
import sys
import pandas as pd
from akarilib import get_colname_lst_of_pixarr
from akarilib import calc_norm_in_row_of_dataframe
from akarilib import calc_stat_in_row_of_dataframe
from akarilib import calc_feature_in_row_of_dataframe
import matplotlib.pyplot as plt

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
print(data_df)

print(data_df.columns)

fig, ax = plt.subplots(1,1)
ax.scatter(data_df["id"], data_df["ti"], s=1, c="black")
ax.set_xlabel("id")
ax.set_ylabel("ti")

outdir = indir
outfile_full = outdir + "/" + "ti_id.png"
print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()
