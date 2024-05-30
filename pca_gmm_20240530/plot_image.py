#
# plot_image.py
#
# plot fits image
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#

import os
import sys
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

#        
# main
#

args = sys.argv
nargs = len(args) - 1
print(nargs)
if (2 == nargs):
    incsv  = args[1]
    outdir = args[2]
    print("incsv = ", incsv)
    print("outdir = ", outdir)
else:
    print('usage: python3  plot_image.py  incsv  outdir')
    print('Arguments are not 2.')
    exit()


# for output
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

# read input
data_df = pd.read_csv(incsv)
print(data_df)

nrow = len(data_df)
for irow in range(nrow):
    filename = data_df.loc[irow, "file"]
    filename_full = os.environ["AKARI_DATA_DIR"] + "/" + filename

    fig, ax = plt.subplots(1, 1)
    hdu = fits.open(filename_full)
    hdu0 = hdu[0]
    ax.imshow(hdu0.data)
    title = (f"{filename}\n")
    ax.set_title(title)

    outfile_full = (outdir + "/" + filename + ".png")
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()
    hdu.close()
