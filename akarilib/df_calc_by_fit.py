import pandas as pd
import numpy as np
from scipy.optimize import curve_fit 
from akarilib import get_colname_lst_of_pixarr_norm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# 1dim gaussian function
def gaussian_1d(in_1darr, mu, sigma, norm):
    out_1darr = (norm * np.exp(
        -1.0 * (in_1darr - mu)**2 / (2 * sigma**2)))
    return out_1darr

# 1dim constant function
def const_1d(in_1darr, const):
    out_1darr = const
    return out_1darr

def gaussian_2d_simple(xyval_2darr, mu_x, mu_y, 
                       sigma_x, sigma_y,
                       norm, const):
    (xval_1darr, yval_1darr) = xyval_2darr
    zval_1darr = (norm / (2.0 * np.pi * sigma_x * sigma_y)
                  * np.exp(-1.0
                           * (xval_1darr - mu_x)**2
                           / (2 * sigma_x**2))
                  * np.exp(-1.0
                           * (yval_1darr - mu_y)**2
                           / (2 * sigma_y**2)) + const)
    #return zval_1darr.ravel()
    return zval_1darr

# 2dim gaussian function
# mu vector: mu_1darr
# covariant matrix: sigma_mat
# 

def gaussian_2d(xyval_2darr, mu_x, mu_y, 
                sigma_x, sigma_y,
                theta, norm, const):
    (xval_1darr, yval_1darr) = xyval_2darr
    rot_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    diag_mat = np.matrix([[1.0/sigma_x**2, 0.0],
                          [0.0, 1.0/sigma_y**2]])
    center_mat = np.matmul(np.matmul(rot_mat, diag_mat), rot_mat.T)

    xyval_vec_2darr = np.array([xval_1darr - mu_x, yval_1darr - mu_y])
    right_vec_2darr = np.matmul(center_mat, xyval_vec_2darr)
    exp_val_1darr = np.diag(np.matmul(xyval_vec_2darr.T, right_vec_2darr))
    zval_1darr = (norm / (2.0 * np.pi * sigma_x * sigma_y)
                  * np.exp( -1.0 * exp_val_1darr / 2.0) + const )
    # return zval_1darr.ravel()
    return zval_1darr

def calc_2dgaussfit_in_row_of_dataframe(row_ser,
                                        xbin_center_2darr,
                                        ybin_center_2darr):
    # call by 
    # data_df.apply(calc_2dgaussfit_in_row_of_dataframe,
    #     args=(xbin_center_2darr, ybin_center_2darr), axis=1)

    xybin_center_2darr = np.array([xbin_center_2darr.ravel(),
                                   ybin_center_2darr.ravel()])
    zval_1darr = row_ser.values

    # for initial vlaue
    zval_nomax_1darr = np.delete(zval_1darr, np.argmax(zval_1darr))
    const_init = np.average(zval_nomax_1darr)

    par_init = np.array([0.0, 0.0, 0.1, 0.1, 1.0,  const_init])
    bounds_simple_tpl = ([-1.0, -1.0,  0.0,  0.0,  0.0, 0.0],
                         [+1.0, +1.0,  0.5,  0.5, 10.0, 1.0])
    valid_2dgauss_simple = 0
    try:
        popt, pcov = curve_fit(gaussian_2d_simple,
                               xybin_center_2darr,
                               zval_1darr,
                               p0=par_init,
                               bounds=bounds_simple_tpl)
        valid_2dgauss_simple = 1
    except:
        popt = np.zeros(6)
        valid_2dgauss_simple = 0

    popt_ser = pd.Series([], dtype=float)
    popt_ser["gfit_simple_mu_x"] = popt[0]
    popt_ser["gfit_simple_mu_y"] = popt[1]
    popt_ser["gfit_simple_sigma_x"] = popt[2]
    popt_ser["gfit_simple_sigma_y"] = popt[3]
    popt_ser["gfit_simple_norm"] = popt[4]
    popt_ser["gfit_simple_const"] = popt[5]
    popt_ser["gfit_simple_valid"] = valid_2dgauss_simple

    par_init = np.array([
        popt_ser["gfit_simple_mu_x"],
        popt_ser["gfit_simple_mu_y"],
        popt_ser["gfit_simple_sigma_x"],
        popt_ser["gfit_simple_sigma_y"],
        0.0,
        popt_ser["gfit_simple_norm"],
        popt_ser["gfit_simple_const"]
    ])
    bounds_tpl = ([-1.0, -1.0,  0.0,  0.0,      0.0,  0.0, 0.0],
                  [+1.0, +1.0,  0.5,  0.5, np.pi/2., 10.0, 1.0])
    valid_2dgauss = 0
    try:
        popt, pcov = curve_fit(gaussian_2d,
                               xybin_center_2darr,
                               zval_1darr,
                               p0=par_init,
                               bounds=bounds_tpl)
        valid_2dgauss = 1
    except:
        popt = np.zeros(8)
        valid_2dgauss = 0

    # poptは最適推定値、pcovは共分散
    popt_ser["gfit_mu_x"] = popt[0]
    popt_ser["gfit_mu_y"] = popt[1]
    popt_ser["gfit_sigma_x"] = popt[2]
    popt_ser["gfit_sigma_y"] = popt[3]
    popt_ser["gfit_theta"] = popt[4]
    popt_ser["gfit_norm"] = popt[5]
    popt_ser["gfit_const"] = popt[6]
    popt_ser["gfit_valid"] = valid_2dgauss
    print(popt_ser)

    return popt_ser


# check result

def check_2dgaussfit_in_row_of_dataframe(row_ser,
                                         xbin_center_2darr,
                                         ybin_center_2darr):
    # call by 
    # data_df.apply(check_2dgaussfit_in_row_of_dataframe,
    #     args=(xbin_center_2darr, ybin_center_2darr), axis=1)

    pixarr_norm_ser = row_ser[get_colname_lst_of_pixarr_norm()]
    zval_1darr = pixarr_norm_ser.values

    # model
    xybin_center_2darr = np.array([xbin_center_2darr.ravel(),
                                   ybin_center_2darr.ravel()])
    model_1darr = gaussian_2d(xybin_center_2darr,
                              row_ser["gfit_mu_x"],
                              row_ser["gfit_mu_y"],
                              row_ser["gfit_sigma_x"],
                              row_ser["gfit_sigma_y"],
                              row_ser["gfit_theta"],
                              row_ser["gfit_norm"],
                              row_ser["gfit_const"])
    residual_1darr = zval_1darr - model_1darr
    print(model_1darr)
    print(zval_1darr)

    fig, ax = plt.subplots(1,2,dpi=140,sharey=True,figsize=(7,4))
    # ax=ax.ravel()
    ax[0].imshow(zval_1darr.reshape(5,5),
                 extent=[-5,5,5,-5],origin="lower")
    ax[1].imshow(model_1darr.reshape(5,5),
                 extent=[-5,5,5,-5],origin="lower")
    outfile_full = ("temp.png")
    plt.savefig(outfile_full,
                bbox_inches='tight',
                pad_inches=0.1)
    print(outfile_full)
    plt.clf()
    plt.close()
    del fig
    del ax

    print(row_ser)
    exit()






