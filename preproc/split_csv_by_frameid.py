#
# split_csv_by_frameid.py
#
# split csv file by frame_id
#

# Preparation:
#   % conda install astropy
#   % conda install scikit-learn
#   % conda install matplotlib
#
# Setup:
#   % source $akari_tool_dir/setup/setup.sh
# Run:
#   % python $akari_tool/preproc_val/fits_to_csv.py
#   % python $akari_tool/preproc_val/plot_ti.py
#   % python $akari_tool/preproc_val/split_csv_by_frameid.py
#

import os
import sys
import pandas as pd

indir = os.environ["AKARI_ANA_DIR"]
incsv = indir + "/" + "akari.csv"
data_df = pd.read_csv(incsv)
nrow = len(data_df)
print(nrow)

# data_id
data_id = os.path.basename(indir)
ana_root_dir = os.path.dirname(indir)

print("data_id = ", data_id)
print("ana_root_dir = ", ana_root_dir)

if ((data_id == "stm13") or (data_id == "stm13r") or
    (data_id == "stm43") or (data_id == "stm43r")):
    nrow_1 = 0
    nrow_2 = 0
    nrow_3 = 0
    data_1_df = data_df[data_df["id"] < 150000]
    nrow_1 = len(data_1_df)

    data_2_df = data_df[(data_df["id"] >= 150000) &
                        (data_df["id"] < 210000)]
    nrow_2 = len(data_2_df)

    data_3_df = data_df[(data_df["id"] >= 210000)]
    nrow_3 = len(data_3_df)

    print("nrow_1 + nrow_2 + nrow_3 = ",
          nrow_1 + nrow_2 + nrow_3)

    # data1
    outdir = ana_root_dir + "/" + data_id + "_1"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_1_df.to_csv(outcsv, index=False)

    # data2
    outdir = ana_root_dir + "/" + data_id + "_2"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_2_df.to_csv(outcsv, index=False)

    # data3
    outdir = ana_root_dir + "/" + data_id + "_3"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_3_df.to_csv(outcsv, index=False)

elif ((data_id == "stm63") or (data_id == "stm63r")):
    nrow_1 = 0
    nrow_2 = 0
    nrow_3 = 0
    nrow_4 = 0
    data_1_df = data_df[data_df["id"] < 30000]
    nrow_1 = len(data_1_df)

    data_2_df = data_df[(data_df["id"] >= 30000) &
                        (data_df["id"] < 100000)]
    nrow_2 = len(data_2_df)

    data_3_df = data_df[(data_df["id"] >= 100000) &
                        (data_df["id"] <  210000)]
    nrow_3 = len(data_3_df)

    data_4_df = data_df[(data_df["id"] >= 210000)]
    nrow_4 = len(data_4_df)

    print("nrow_1 + nrow_2 + nrow_3 + nrow_4 = ",
          nrow_1 + nrow_2 + nrow_3 + nrow_4)

    # data1
    outdir = ana_root_dir + "/" + data_id + "_1"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_1_df.to_csv(outcsv, index=False)

    # data2
    outdir = ana_root_dir + "/" + data_id + "_2"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_2_df.to_csv(outcsv, index=False)

    # data3
    outdir = ana_root_dir + "/" + data_id + "_3"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_3_df.to_csv(outcsv, index=False)

    # data4
    outdir = ana_root_dir + "/" + data_id + "_4"
    if (False == os.path.exists(outdir)):
        os.makedirs(outdir)
    outcsv = outdir + "/" + "akari.csv"
    print(f"outcsv = {outcsv}")
    data_4_df.to_csv(outcsv, index=False)


else:
    print("not supported.")
    exit()

