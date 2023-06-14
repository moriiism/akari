import pandas as pd
from akarilib import gini
from astropy import units
from astropy.coordinates import angular_separation, Angle


def calc_feature_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(
    #   calc_feature_in_row_of_dataframe, axis=1)
    row_sel_ser = row_ser.drop("sum")
    row_feature_ser = pd.Series([], dtype=float)
    peak = row_sel_ser["x02y02_norm"]
    around = (row_sel_ser["x01y01_norm"]
              + row_sel_ser["x01y02_norm"]
              + row_sel_ser["x01y03_norm"]
              + row_sel_ser["x02y01_norm"]
              + row_sel_ser["x02y03_norm"]
              + row_sel_ser["x03y01_norm"]
              + row_sel_ser["x03y02_norm"]
              + row_sel_ser["x03y03_norm"]) / 8.0
    ratio_around_to_peak = around / peak
    row_feature_ser["ratio_around_to_peak"] = ratio_around_to_peak
    return row_feature_ser

def calc_norm_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(calc_norm_in_row_of_dataframe, axis=1)
    row_norm_ser = row_ser.copy()
    total = row_ser.sum()
    index_norm_idx = row_ser.index + "_norm"
    row_norm_ser = row_ser / total
    row_norm_ser.index = index_norm_idx
    row_norm_ser["sum"] = total
    return row_norm_ser

def calc_stat_in_row_of_dataframe(row_ser):
    # call by data_frame.apply(calc_stat_in_row_of_dataframe, axis=1)
    row_sel_ser = row_ser.drop("sum")
    row_stat_ser = pd.Series([], dtype=float)
    row_stat_ser["norm_stddev"] = row_sel_ser.std()
    row_stat_ser["norm_min"] = row_sel_ser.min()
    row_stat_ser["norm_max"] = row_sel_ser.max()
    row_stat_ser["norm_skew"] = row_sel_ser.skew()
    row_stat_ser["norm_kurt"] = row_sel_ser.kurt()
    row_stat_ser["norm_gini"] = gini(row_sel_ser.values)
    return row_stat_ser

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



