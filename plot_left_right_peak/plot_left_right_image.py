#
# plot_left_right_image.py
#
# plot fits image (left, right)
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


def run_plot_left_right_image(list_file, out_subdir):

    data_dir = os.environ["AKARI_DATA_DIR"]
    indir = os.environ["AKARI_ANA_DIR"]
    ana_dir = os.environ["AKARI_ANA_DIR"]
    list_file_full = ana_dir + "/" + list_file
    outdir = indir + "/" + out_subdir
    os.mkdir(outdir)

    with open(list_file_full) as fptr:
        for file_name_2 in fptr:
            print(file_name_2)
            (file_name_left, x_left, y_left, peak_left,
             file_name_right, x_right, y_right, peak_right,
             nstar_cat8) = file_name_2.split()
            file_name_left_full = data_dir + "/" + file_name_left
            file_name_right_full = data_dir + "/" + file_name_right
            print(file_name_left_full)
            print(file_name_right_full)
    
            fig, ax = plt.subplots(1, 2)
    
            hdu = fits.open(file_name_left_full)
            hdu0 = hdu[0]
            ax[0].imshow(hdu0.data)
            title_left = (f"{file_name_left}\n"
                          + f"peak={peak_left}, nstar_cat8={nstar_cat8}")
            xlabel_left = f"tzl_x = {x_left}"
            ylabel_left = f"tzl_y = {y_left}"
            ax[0].set_title(title_left)
            ax[0].set_xlabel(xlabel_left)
            ax[0].set_ylabel(ylabel_left)
    
            hdu = fits.open(file_name_right_full)
            hdu0 = hdu[0]
            ax[1].imshow(hdu0.data)
            title_right = f"{file_name_right}\n peak={peak_right}"
            xlabel_right = f"tzl_x = {x_right}"
            ylabel_right = f"tzl_y = {y_right}"
            ax[1].set_title(title_right)
            ax[1].set_xlabel(xlabel_right)
            ax[1].set_ylabel(ylabel_right)
    
            outfile_full = (outdir + "/"
                            + file_name_left + "_" + file_name_right 
                            + ".png")
            print("outfile = ", outfile_full)
            plt.savefig(outfile_full,
                        bbox_inches='tight',
                        pad_inches=0.1)
            plt.cla()
            plt.clf()
            plt.close()

# main

list_file = "peak_left_right_file_pm2pix_cluster.list"
out_subdir = "peak_left_right_file_pm2pix_cluster"
run_plot_left_right_image(list_file, out_subdir)

#list_file = "peak_left_right_file_pm2pix.list"
#out_subdir = "peak_left_right_file_pm2pix"
#run_plot_left_right_image(list_file, out_subdir)
#
#list_file = "peak_left_right_file_pm1pix.list"
#out_subdir = "peak_left_right_file_pm1pix"
#run_plot_left_right_image(list_file, out_subdir)
#
#list_file = "peak_left_right_file_cat.list"
#out_subdir = "peak_left_right_file_cat"
#run_plot_left_right_image(list_file, out_subdir)
#
