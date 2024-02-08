#!/bin/sh

export PCA_MODEL=/home/morii/work/akari/ana/st20070102A2/pca.pkl
export GMM_MODEL=/home/morii/work/akari/ana/st20070102A2/gmm.pkl
echo "PCA_MODEL="$PCA_MODEL
echo "GMM_MODEL="$GMM_MODEL

export PCA_MARGIN_MODEL=/home/morii/work/akari/ana/st20070102A2/pca_margin.pkl
export GMM_MARGIN_MODEL=/home/morii/work/akari/ana/st20070102A2/gmm_margin.pkl
echo "PCA_MARGIN_MODEL="$PCA_MARGIN_MODEL
echo "GMM_MARGIN_MODEL="$GMM_MARGIN_MODEL


echo "load_prefit pca gmm models"
