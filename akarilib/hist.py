#
# hist.py
#

import numpy as np

# generate meshgrid of xy bin center
def get_meshgrid_xybin_center(nbinx: int, 
                              lo_xbin_center: float, 
                              up_xbin_center: float,
                              nbiny: int, 
                              lo_ybin_center: float, 
                              up_ybin_center: float):
    # 2-tuple of numpy 2darray-s: (xbin_center_2darr, ybin_center_2darr).
    # xbin_center_2darr (ybin_center_2darr) is a numpy 2darray of x-axis
    # (y-axis) values of a 2-dim histogram.
    # 
    xbin_center_1darr = np.linspace(lo_xbin_center,
                                    up_xbin_center,
                                    nbinx)
    ybin_center_1darr = np.linspace(lo_ybin_center,
                                    up_ybin_center,
                                    nbiny)
    (xbin_center_2darr, ybin_center_2darr) = np.meshgrid(
        xbin_center_1darr, ybin_center_1darr)

    return (xbin_center_2darr, ybin_center_2darr)


#def hist1d(1d_arr):
#    
#    return 
#
#def hist2d(2d_arr):
#    H = ax.hist2d(x,y, bins=40, cmap=cm.jet)
#
