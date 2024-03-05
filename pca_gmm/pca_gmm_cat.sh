#!/bin/bash

# for data with catalog to make model

cd /home/morii/work/github/moriiism/akari/pca_gmm
source ../setup/setup_st20070102A2.sh

flag_cat=1

###
use_prefit=0
pca_feature=0
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=1
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=2
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=3
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=4
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def


###
use_prefit=1
pca_feature=0
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=1
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=2
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=3
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
pca_feature=4
python3 add_flag_pca.py $flag_cat $pca_feature $use_prefit def def def def
python3 gmm.py $flag_cat $pca_feature $use_prefit def def def def
