import pandas as pd
from akarilib import gini

def calcNormInRowOfDataFrame(row_ser):
    # call by data_frame.apply(calcNormInRowOfDataFrame, axis=1)
    row_norm_ser = row_ser.copy()
    total = row_ser.sum()
    index_norm_idx = row_ser.index + "_norm"
    row_norm_ser = row_ser / total
    row_norm_ser.index = index_norm_idx
    row_norm_ser["sum"] = total
    return row_norm_ser

def calcStatInRowOfDataFrame(row_ser):
    # call by data_frame.apply(calcStatInRowOfDataFrame, axis=1)
    row_sel_ser = row_ser.drop("sum")
    row_stat_ser = pd.Series([], dtype=float)
    row_stat_ser["norm_stddev"] = row_sel_ser.std()
    row_stat_ser["norm_min"] = row_sel_ser.min()
    row_stat_ser["norm_max"] = row_sel_ser.max()
    row_stat_ser["norm_skew"] = row_sel_ser.skew()
    row_stat_ser["norm_kurt"] = row_sel_ser.kurt()
    row_stat_ser["norm_gini"] = gini(row_sel_ser.values)
    
    return row_stat_ser

