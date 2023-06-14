#
# add_flag_star_pos.py
#
# add flag of star/spike by position
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
#   % python $akari_tool/preproc/add_stat.py
#   % python $akari_tool/preproc/add_stat_fit.py
#   % python $akari_tool/preproc/add_flag_star_pos.py
#

import os
import sys
import pandas as pd
from akarilib import calc_norm_in_row_of_dataframe
from akarilib import calc_stat_in_row_of_dataframe

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari_stat_fit.csv"
data_df = pd.read_csv(incsv)
print(data_df)
nrow = len(data_df)

data_sel_df = data_df[["file", "tz_x", "tz_y", "tzl_x", "tzl_y"]]
flag_df = pd.DataFrame([], index=range(nrow),
                       columns=["nfind",
                                "file_find", 
                                "star_pos",
                                "diff_tzl_x",
                                "diff_tzl_y"])
flag_df.loc[:, "nfind"] = 0
flag_df.loc[:, "file_find"] = ""
flag_df.loc[:, "star_pos"] = 0
flag_df.loc[:, "diff_tzl_x"] = 0
flag_df.loc[:, "diff_tzl_y"] = 0
flag_df["nfind"] = flag_df["nfind"].astype(int)
flag_df["star_pos"] = flag_df["star_pos"].astype(int)
flag_df["diff_tzl_x"] = flag_df["diff_tzl_x"].astype(int)
flag_df["diff_tzl_y"] = flag_df["diff_tzl_y"].astype(int)

# loop for left hand side
for irow1 in range(len(data_sel_df)):
    tz_x = int(data_sel_df.loc[irow1, "tz_x"])
    # skip right hand side image
    if tz_x >= 64:
        continue
    file1 = data_sel_df.loc[irow1, "file"]
    tzl_x_1 = int(data_sel_df.loc[irow1, "tzl_x"])
    tzl_y_1 = int(data_sel_df.loc[irow1, "tzl_y"])
    file1_split = file1.split(".")
    (obsid1, dummy, sernum1) = file1_split[0].split("_")

    # loop for right hand side
    nfind = 0
    file_find_lst = []
    for irow2 in range(len(data_sel_df)):
        tz_x = int(data_sel_df.loc[irow2, "tz_x"])
        # skip left hand side image
        if tz_x < 64:
            continue
        file2 = data_sel_df.loc[irow2, "file"]
        tzl_x_2 = int(data_sel_df.loc[irow2, "tzl_x"])
        tzl_y_2 = int(data_sel_df.loc[irow2, "tzl_y"])
        file2_split = file2.split(".")
        (obsid2, dummy, sernum2) = file2_split[0].split("_")

        diff_tzl_x = tzl_x_2 - tzl_x_1
        diff_tzl_y = tzl_y_2 - tzl_y_1
        if( (obsid1 == obsid2) &
            (sernum1 != sernum2) ):
            star_pos = 0
            if ( ( abs(tzl_x_1 - tzl_x_2) < 3 ) &
                 ( abs(tzl_y_1 - tzl_y_2) < 3 ) ):
                star_pos = 1
            else:
                pass
            if ( ( abs(tzl_x_1 - tzl_x_2) < 2 ) &
                 ( abs(tzl_y_1 - tzl_y_2) < 2 ) ):
                star_pos += 1
            else:
                pass
            if ( ( abs(tzl_x_1 - tzl_x_2) < 1 ) &
                 ( abs(tzl_y_1 - tzl_y_2) < 1 ) ):
                star_pos += 1
            else:
                pass
                
            if (star_pos > 0):
                nfind += 1
                flag_df.loc[irow1, "nfind"] = nfind
                if (star_pos > flag_df.loc[irow1, "star_pos"]):
                    flag_df.loc[irow1, "star_pos"] = star_pos
                    flag_df.loc[irow1, "file_find"] = file2
                    flag_df.loc[irow1, "diff_tzl_x"] = diff_tzl_x
                    flag_df.loc[irow1, "diff_tzl_y"] = diff_tzl_y

                file_find_lst.append(file2)

    if nfind > 0:
        print(file_find_lst)

data_add_df = pd.concat([data_df, flag_df], axis=1)

print(data_add_df[["star_pos"]].value_counts())
print(data_add_df[["nfind"]].value_counts())
print(data_add_df[["nfind", "star_pos"]].value_counts())



outdir = indir
outcsv = outdir + "/" + "akari_stat_fit_star.csv"
print(f"outcsv = {outcsv}")
data_add_df.to_csv(outcsv, index=False)
