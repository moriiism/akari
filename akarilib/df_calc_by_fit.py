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


def gaussian_2d_simple(xyval, mu_x, mu_y, sigma_x, sigma_y,
                       norm, const):
    (xval, yval) = xyval
    zval_2darr = (norm / (2.0 * np.pi * sigma_x * sigma_y)
                  * np.exp(-1.0
                           * (xval - mu_x)**2
                           / (2 * sigma_x**2))
                  * np.exp(-1.0
                           * (yval - mu_y)**2
                           / (2 * sigma_y**2)) + const)
    return zval_2darr.ravel()

# 2dim gaussian function
# mu vector: mu_1darr
# covariant matrix: sigma_mat
# 

def gaussian_2d(xyval, mu_x, mu_y, sigma_x, sigma_y,
                theta, norm, const):
    (xval, yval) = xyval
    rot_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    diag_mat = np.matrix([[1.0/sigma_x**2, 0.0],
                          [0.0, 1.0/sigma_y**2]])
    center_mat = np.matmul(np.matmul(rot_mat, diag_mat), rot_mat.T)

    xyval_vec = np.array([xval - mu_x, yval - mu_y])
    right_vec = np.matmul(center_mat, xyval_vec)
    exp_val_1darr = np.diag(np.matmul(xyval_vec.T, right_vec))
    zval_2darr = (norm / (2.0 * np.pi * sigma_x * sigma_y)
                  * np.exp( -1.0 * exp_val_1darr / 2.0) + const )

    #    zval_2darr = (norm
    #                  * np.exp(-1.0
    #                           * (xval - mu_x)**2
    #                           / (2 * sigma_x**2))
    #                  * np.exp(-1.0
    #                           * (yval - mu_y)**2
    #                           / (2 * sigma_y**2)) + const)
    return zval_2darr.ravel()

def calc_2dgaussfit_in_row_of_dataframe(row_ser,
                                        xbin_center_2darr,
                                        ybin_center_2darr):
    # call by 
    # data_df.apply(calc_2dgaussfit_in_row_of_dataframe,
    #     args=(xbin_center_2darr, ybin_center_2darr), axis=1)

    xybin_center_2darr = np.array([xbin_center_2darr.ravel(),
                                   ybin_center_2darr.ravel()])
    bounds_tpl = ([-1.0, -1.0,  0.0,  0.0,  0.0, 0.0],
                  [+1.0, +1.0, 10.0, 10.0, 10.0, 1.0])

    popt, pcov = curve_fit(gaussian_2d_simple,
                           xybin_center_2darr,
                           zval_2darr,
                           bounds=bounds_tpl)
    # poptは最適推定値、pcovは共分散
    popt_ser = pd.Series([], dtype=float)
    popt_ser["gfit_mu_x"] = popt[0]
    popt_ser["gfit_mu_y"] = popt[1]
    popt_ser["gfit_sigma_x"] = popt[2]
    popt_ser["gfit_sigma_y"] = popt[3]
    popt_ser["gfit_theta"] = popt[4]
    popt_ser["gfit_norm"] = popt[5]
    popt_ser["gfit_const"] = popt[6]
    print(popt_ser)



    zval_2darr = row_ser.values

    par_init = np.array([])
    xybin_center_2darr = np.array([xbin_center_2darr.ravel(),
                                   ybin_center_2darr.ravel()])
    popt, pcov = curve_fit(gaussian_2d,
                           xybin_center_2darr,
                           zval_2darr,
                           p0=par_init,
                           bounds=bounds_tpl)
    # poptは最適推定値、pcovは共分散
    popt_ser = pd.Series([], dtype=float)
    popt_ser["gfit_mu_x"] = popt[0]
    popt_ser["gfit_mu_y"] = popt[1]
    popt_ser["gfit_sigma_x"] = popt[2]
    popt_ser["gfit_sigma_y"] = popt[3]
    popt_ser["gfit_theta"] = popt[4]
    popt_ser["gfit_norm"] = popt[5]
    popt_ser["gfit_const"] = popt[6]
    print(popt_ser)

    return popt_ser
