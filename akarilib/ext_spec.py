from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

# from akarilib import get_filter_np_or_ng

def get_filter_np_or_ng(hdu0):
    # AOTPARAM= 'a;Ns    '
    aot_param_char = hdu0.header["AOTPARAM"][0]
    # a: prism, b: grism
    filter = ""
    if ("a" == aot_param_char):
        filter = "np"
    elif ("b" == aot_param_char):
        filter = "ng"
    else:
        print("error")
        exit()

    return(filter)

def get_pixel_loup(filter, region_id):
    xlo = 0
    xup = 0
    ylo = 0
    yup = 0
    if ("np" == filter):
        if (1 == region_id):
            xlo = 10
            xup = 30 - 1
            ylo = 200
            yup = 300 - 1
        elif (2 == region_id):
            xlo = 95
            xup = 105 - 1
            ylo = 200
            yup = 300 - 1
        else:
            print("error")
            exit()
    elif ("ng" == filter):
        if (1 == region_id):
            xlo = 10
            xup = 30 - 1
            ylo = 0
            yup = 412 - 1
        elif (2 == region_id):
            xlo = 95
            xup = 105 - 1
            ylo = 0
            yup = 412 - 1
        else:
            print("error")
            exit()
    else:
        print("error")
        exit()

    loup_lst = [xlo, xup, ylo, yup]
    return(loup_lst)

def get_npix_rebin_lst(region_id):
    npix_rebin_lst = []
    if (1 == region_id):
        # 20 = 1 x 20
        #    = 2 x 10
        #    = 4 x 5
        #    = 5 x 4
        #    = 10 x 2
        #    = 20 x 1
        npix_rebin_lst = [1,2,4,5,10,20]
    elif (2 == region_id):
        # 10 = 1 x 10
        #    = 2 x 5
        #    = 5 x 2
        #    = 10 x 1
        npix_rebin_lst = [1,2,5,10]
    else:
        print("error")
        exit()
    return(npix_rebin_lst)

def get_rebin_xloup_lst(xlo, xup, npix_rebin):
    xloup_lst = []
    for ix in range(xlo, xup + 1, npix_rebin):
        xloup_lst.append([ix, ix + npix_rebin - 1])
    return(xloup_lst) 


def extract_spectrum_from_img_2darr(img_2darr, outdir,
                                    index_zaxis, region_id, npix_rebin,
                                    xlo, xup, ylo, yup):
    print(img_2darr[xlo:xup+1,ylo:yup+1].shape)
    mean_1darr = img_2darr[xlo:xup+1,ylo:yup+1].mean(axis=0)
    index_arr = np.arange(ylo, yup+1)
    stddev_1darr = np.zeros(index_arr.size)
    if (npix_rebin >= 2):
        stddev_1darr = img_2darr[xlo:xup+1,ylo:yup+1].std(ddof=1, axis=0)
        stddev_1darr /= np.sqrt(npix_rebin)
    plt.errorbar(index_arr, mean_1darr, yerr=stddev_1darr,
                 marker='o', capthick=1, lw=1)
    outfile_full = (outdir + "/" + 
                    "spec_z%1.1d_reg%1.1d_rebin%2.2d_%3.3d-%3.3d.png" % (
                        index_zaxis, region_id, npix_rebin,
                        xlo, xup))
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.cla()
    plt.clf()
    plt.close()

def extract_spectrum_from_fits(infile, outdir):
    # image size: (512, 412)
    hdu = fits.open(infile)
    hdu0 = hdu[0] 
    filter = get_filter_np_or_ng(hdu0)

    # index_zaxis: index of cube data z-axis
    for index_zaxis in [0, 1]:
        print("index_zaxis = ", index_zaxis)
        img_2darr = hdu0.data[index_zaxis].T
        print(img_2darr.shape)
        for region_id in [1, 2]:
            print("region_id = ", region_id)
            (xlo, xup, ylo, yup) = get_pixel_loup(filter, region_id)
            print("xlo, xup, ylo, yup = ", xlo, xup, ylo, yup)
            npix_rebin_lst = get_npix_rebin_lst(region_id)
            print("npix_rebin_lst = ", npix_rebin_lst)
            for npix_rebin in npix_rebin_lst:
                xloup_lst = get_rebin_xloup_lst(xlo, xup, npix_rebin)
                for xloup in xloup_lst:
                    (xlo_this, xup_this) = xloup
                    extract_spectrum_from_img_2darr(img_2darr, outdir,
                                                    index_zaxis, region_id,
                                                    npix_rebin,
                                                    xlo_this, xup_this,
                                                    ylo, yup)

