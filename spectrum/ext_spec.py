#
# ext_spec.py
#
# extract spectrum
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
#

import os
import sys
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

#from akarilib import get_colname_lst_of_pixarr
#from akarilib import ecliptic_to_radec

# argment
infile = ""
xlo = 0
xup = 0
ylo = 0
yup = 0
outdir = ""

args = sys.argv
nargs = len(args) - 1
print("nargs = ", nargs)
if (6 == nargs):
    infile = args[1]
    xlo = int(args[2])
    xup = int(args[3])
    ylo = int(args[4])
    yup = int(args[5])
    outdir = args[6]
    print("infile = ", infile)
    print("xlo xup ylo yup = ", xlo, xup, ylo, yup)
    print("outdir = ", outdir)
    pass
else:
    print('usage: python ext_spec.py ' +
          'infile xlo xup ylo yup outdir')
    print('Arguments are not 6.')
    exit()

#data_dir = os.environ["AKARI_DATA_DIR"]
#
#data_id = "1501633.1"
#
#data_file = (data_dir + "/"
#             + "DATA" + "/"
#             + data_id + "/"
#             + "irc_specred_out"
#             + "/" + "1501633.1.N3_NP.specimage_bg.fits")

fig, ax = plt.subplots(1, 2)

hdu = fits.open(infile)
hdu0 = hdu[0]
ax[0].imshow(hdu0.data[0])
ax[1].imshow(hdu0.data[1])

if (os.path.isdir(outdir) == False):
    os.makedirs(outdir)

outfile_full = (outdir + "/" + "img2.png")

print("outfile = ", outfile_full)
plt.savefig(outfile_full,
            bbox_inches='tight',
            pad_inches=0.1)
plt.cla()
plt.clf()
plt.close()

# hdu0.data[0].shape: (412, 512)
# ds9 image: (512, 412)

img_2darr = hdu0.data[1].T
print(img_2darr.shape)

for npix_rebin in [1, 2, 4, 8, 16]:
    outdir_this = (outdir + "/" + "rebin_%2.2d" % npix_rebin)
    if (os.path.isdir(outdir_this) == False):
        os.mkdir(outdir_this)

    for ix in range(xlo, xup, npix_rebin):
        print("ix = ", ix)
        print(img_2darr[ix:ix+npix_rebin,ylo:yup].shape)
        mean_1darr = img_2darr[ix:ix+npix_rebin,ylo:yup].mean(axis=0)
        index_arr = np.arange(ylo, yup)
        stddev_1darr = np.zeros(index_arr.size)
        if (npix_rebin >= 2):
            stddev_1darr = img_2darr[ix:ix+npix_rebin,ylo:yup].std(ddof=1, axis=0)
            stddev_1darr /= np.sqrt(npix_rebin)
        plt.errorbar(index_arr, mean_1darr, yerr=stddev_1darr,
                     marker='o', capthick=1, lw=1)
        outfile_full = (outdir_this + "/" + 
                        "spec_rebin%2.2d_%3.3d.png" % (npix_rebin, ix))
        print("outfile = ", outfile_full)
        plt.savefig(outfile_full,
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.cla()
        plt.clf()
        plt.close()
