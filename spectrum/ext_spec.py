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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from akarilib import extract_spectrum_from_fits

# argment
infile = ""
outdir = ""

args = sys.argv
nargs = len(args) - 1
print("nargs = ", nargs)
if (2 == nargs):
    infile = args[1]
    outdir = args[2]
    print("infile = ", infile)
    print("outdir = ", outdir)
    pass
else:
    print('usage: python ext_spec_tmp.py ' +
          'infile  outdir')
    print('Arguments are not 2.')
    exit()

if (os.path.isdir(outdir) == False):
    os.mkdir(outdir)

extract_spectrum_from_fits(infile, outdir)
