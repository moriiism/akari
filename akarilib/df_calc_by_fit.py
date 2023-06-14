import pandas as pd
import numpy as np
from scipy.optimize import curve_fit  

# 1dim gaussian function
def gaussian_1d(in_1darr, mu, sigma, norm):
    out_1darr = (norm * np.exp(
        -1.0 * (in_1darr - mu)**2 / (2 * sigma**2)))
    return out_1darr

# 1dim constant function
def const_1d(in_1darr, const):
    out_1darr = const
    return out_1darr

# 2dim gaussian function
# mu vector: mu_1darr
# covariant matrix: sigma_mat
# 

def gaussian_2d(xy_lst, mu_x, mu_y, sigma_x, sigma_y, norm, const):
    (xval_2darr, yval_2darr) = xy_lst
    zval_2darr = (norm
                  * np.exp(-1.0
                           * (xval_2darr - mu_x)**2
                           / (2 * sigma_x**2))
                  * np.exp(-1.0
                           * (yval_2darr - mu_y)**2
                           / (2 * sigma_y**2)) + const)
    return zval_2darr.ravel()

def calc_2dgaussfit_in_row_of_dataframe(row_ser,
                                        xbin_center_2darr,
                                        ybin_center_2darr):
    # call by 
    # data_df.apply(calc_2dgaussfit_in_row_of_dataframe,
    #     args=(xbin_center_2darr, ybin_center_2darr), axis=1)

    zval_2darr = row_ser.values
    bounds_tpl = ([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                  [+1.0, +1.0, 10.0, 10.0, 1e10, 1e10])
    popt, pcov = curve_fit(gaussian_2d,
                           (xbin_center_2darr,
                            ybin_center_2darr),
                           zval_2darr,
                           bounds=bounds_tpl)
    # poptは最適推定値、pcovは共分散

    popt_ser = pd.Series([], dtype=float)
    popt_ser["gfit_mu_x"] = popt[0]
    popt_ser["gfit_mu_y"] = popt[1]
    popt_ser["gfit_sigma_x"] = popt[2]
    popt_ser["gfit_sigma_y"] = popt[3]
    popt_ser["gfit_norm"] = popt[4]
    popt_ser["gfit_const"] = popt[5]
    print(popt_ser)

    return popt_ser
