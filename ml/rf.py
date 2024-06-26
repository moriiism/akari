#
# rf.py
#

import sys
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import mixture
import pickle as pk

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from akarilib import get_colname_lst_of_pixarr_norm


# python3 rf.py

#args = sys.argv
#nargs = len(args) - 1
#print(nargs)
#if (0 == nargs):
#    exit()



def output_graphs(clf, X_test, y_test, feature_lst, outdir):

    y_pred_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0])

    # Feature Importance
    importances = pd.DataFrame({'Importance':clf.feature_importances_}, index=feature_lst)
    importances.sort_values('Importance', ascending=False).head(10).sort_values(
        'Importance', ascending=True).plot.barh(ax=axes[0, 1], grid=True)

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 0])
    axes[1, 0].set_title('ROC(Receiver Operating Characteristic) Curve')

    # 適合率-再現率
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 1])

    # plt.show()

    # plot
    outfile_full = outdir + "/" + "rf_report.png"
    print("outfile = ", outfile_full)
    plt.savefig(outfile_full,
                bbox_inches='tight',  pad_inches=0.1)


#### main

indir = os.environ["AKARI_ANA_DIR"]
incsv = (indir + "/" + "akari_stat_fit_star_cat.csv")

# for output
outdir = indir + "/" + "rf"
if (False == os.path.exists(outdir)):
    os.makedirs(outdir)

# read input
data_df = pd.read_csv(incsv)
print(data_df)

data_left_df = data_df[(data_df["left"] == 1) & 
                       (data_df["dark"] == 0) &
                       (data_df["edge"] == 0) &
                       (data_df["star_pos"] > 1)]
data_left_df.reset_index(inplace=True, drop=True)
data_right_df = data_df[(data_df["left"] == 0) & 
                        (data_df["dark"] == 0) &
                        (data_df["edge"] == 0) &
                        (data_df["star_pos"] > 1)]
data_right_df.reset_index(inplace=True, drop=True)

data_left_df.loc[:,"state"] = (data_left_df["nstar_cat8"]>0)
print(data_left_df["state"].value_counts())

colname_pixarr_norm_lst = get_colname_lst_of_pixarr_norm()
feature_lst = (
    colname_pixarr_norm_lst + ["sum",
                               "norm_stddev",
                               "norm_min",
                               "norm_max",
                               "norm_skew",
                               "norm_kurt",
                               "norm_gini",
                               "ratio_around_to_peak",
                               "gfit_mu_x",
                               "gfit_mu_y",
                               "gfit_sigma_x",
                               "gfit_sigma_y",
                               "gfit_theta",
                               "gfit_norm",
                               "gfit_const",
                               "gfit_valid"])
colname_lst = feature_lst + ["state"]

data_rf_df = data_left_df[colname_lst]
print(data_rf_df.columns)
print(data_rf_df.shape)

train_data = data_rf_df.drop("state", axis=1)
y = data_rf_df["state"].values
X = train_data.values

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.3, random_state=1234)

clf = RandomForestClassifier(random_state=1234)
# learning
clf.fit(X_train, y_train)
print("score=", clf.score(X_test, y_test))


output_graphs(clf, X_test, y_test, feature_lst, outdir)


