import pandas as pd
import numpy as np
from akarilib import gini
from astropy import units
from astropy.coordinates import angular_separation, Angle

def calc_norm_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(calc_norm_in_row_of_dataframe, axis=1)
    row_norm_ser = row_ser.copy()
    total = row_ser.sum()
    index_norm_idx = row_ser.index + "_norm"
    row_norm_ser = row_ser / total
    row_norm_ser.index = index_norm_idx
    row_norm_ser["sum"] = total
    return row_norm_ser

def calc_stat_for_normed_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(calc_stat_for_normed_in_row_of_dataframe,
    #                          axis=1)
    row_sel_ser = row_ser.drop("sum")
    row_stat_ser = pd.Series([], dtype=float)
    row_stat_ser["norm_stddev"] = row_sel_ser.std()
    row_stat_ser["norm_min"] = row_sel_ser.min()
    row_stat_ser["norm_max"] = row_sel_ser.max()
    row_stat_ser["norm_skew"] = row_sel_ser.skew()
    row_stat_ser["norm_kurt"] = row_sel_ser.kurt()
    row_stat_ser["norm_gini"] = gini(row_sel_ser.values)
    return row_stat_ser

def calc_feature_for_normed_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(
    #   calc_feature_for_normed_in_row_of_dataframe, axis=1)
    row_feature_ser = pd.Series([], dtype=float)
    peak = row_ser["x02y02_norm"]
    around = (row_ser["x01y01_norm"]
              + row_ser["x01y02_norm"]
              + row_ser["x01y03_norm"]
              + row_ser["x02y01_norm"]
              + row_ser["x02y03_norm"]
              + row_ser["x03y01_norm"]
              + row_ser["x03y02_norm"]
              + row_ser["x03y03_norm"]) / 8.0
    ratio_around_to_peak = around / peak
    row_feature_ser["ratio_around_to_peak"] = ratio_around_to_peak
    norm_ave_margin = (row_ser["x00y00_norm"]
                       + row_ser["x00y01_norm"]
                       + row_ser["x00y02_norm"]
                       + row_ser["x00y03_norm"]
                       + row_ser["x00y04_norm"]
                       + row_ser["x01y00_norm"]
                       + row_ser["x01y04_norm"]
                       + row_ser["x02y00_norm"]
                       + row_ser["x02y04_norm"]
                       + row_ser["x03y00_norm"]
                       + row_ser["x03y04_norm"]
                       + row_ser["x04y00_norm"]
                       + row_ser["x04y01_norm"]
                       + row_ser["x04y02_norm"]
                       + row_ser["x04y03_norm"]
                       + row_ser["x04y04_norm"]) / 16.0
    row_feature_ser["norm_ave_margin"] = norm_ave_margin
    return row_feature_ser


def calc_feature_for_asis_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(
    #   calc_feature_for_asis_in_row_of_dataframe, axis=1)
    row_feature_ser = pd.Series([], dtype=float)
    ave_around = (row_ser["x01y01"]
                  + row_ser["x01y02"]
                  + row_ser["x01y03"]
                  + row_ser["x02y01"]
                  + row_ser["x02y03"]
                  + row_ser["x03y01"]
                  + row_ser["x03y02"]
                  + row_ser["x03y03"]) / 8.0
    row_feature_ser["ave_around"] = ave_around
    ave_margin = (row_ser["x00y00"]
                  + row_ser["x00y01"]
                  + row_ser["x00y02"]
                  + row_ser["x00y03"]
                  + row_ser["x00y04"]
                  + row_ser["x01y00"]
                  + row_ser["x01y04"]
                  + row_ser["x02y00"]
                  + row_ser["x02y04"]
                  + row_ser["x03y00"]
                  + row_ser["x03y04"]
                  + row_ser["x04y00"]
                  + row_ser["x04y01"]
                  + row_ser["x04y02"]
                  + row_ser["x04y03"]
                  + row_ser["x04y04"]) / 16.0
    row_feature_ser["ave_margin"] = ave_margin
    return row_feature_ser


def calc_angular_separation_in_row_of_dataframe(
        row_ser, ra, dec):
    # call by data_frame.apply(
    #   calc_angular_separation_in_row_of_dataframe(ra, dec), axis=1)

    ra_cat = row_ser["ra_cat"]
    dec_cat = row_ser["dec_cat"]
    separation = angular_separation(
        Angle(ra, units.degree),
        Angle(dec, units.degree),
        Angle(ra_cat, units.degree),
        Angle(dec_cat, units.degree))
    print(separation)
    if (separation.to(units.arcsec).value < 5):
        return 1
    else:
        return 0


def calc_dist_lr_for_pca_in_row_of_dataframe(row_ser,
                                             right_df):
    # call by data_frame.apply(
    #    calc_dist_lr_for_pca_in_row_of_dataframe(right_df), axis=1)

    left_1darr = row_ser.filter(regex="sum").values
    row_dist_ser = pd.Series([], dtype=float)
    row_dist_ser["dist_lr_pca"] = -1.0
    if (row_ser["nfind"]==0):
        row_dist_ser["dist_lr_pca"] = -1.0
    elif (row_ser["nfind"]>0):
        right_match_df = None
        right_match_df = right_df[right_df["file"]==row_ser["file_find"]]
        if (len(right_match_df) == 0):
            row_dist_ser["dist_lr_pca"] = -1.0
        else:
            right_1darr = right_match_df.iloc[0,:].filter(regex="sum").values
            row_dist_ser["dist_lr_pca"] = np.linalg.norm(
                left_1darr - right_1darr)
        del right_match_df

    return (row_dist_ser)


#def calc_dist_lr_for_pca_in_row_of_dataframe(row_ser,
#                                             right_df):
#    # call by data_frame.apply(
#    #    calc_dist_lr_for_pca_in_row_of_dataframe(right_df), axis=1)
#
#    left_1darr = row_ser.filter(regex="^pc").values
#    row_dist_ser = pd.Series([], dtype=float)
#    row_dist_ser["dist_lr_pca"] = -1.0
#    if (row_ser["nfind"]==0):
#        row_dist_ser["dist_lr_pca"] = -1.0
#    elif (row_ser["nfind"]>0):
#        right_match_df = None
#        right_match_df = right_df[right_df["file"]==row_ser["file_find"]]
#        if (len(right_match_df) == 0):
#            row_dist_ser["dist_lr_pca"] = -1.0
#        else:
#            right_1darr = right_match_df.iloc[0,:].filter(regex="^pc").values
#            row_dist_ser["dist_lr_pca"] = np.linalg.norm(
#                left_1darr - right_1darr)
#        del right_match_df
#
#    return (row_dist_ser)




    
