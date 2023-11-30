#!/bin/sh

reg2_corner="45 80 190 260"

python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
1501633.1/irc_specred_out/1501633.1.N3_NP.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/1501633.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
1501662.1/irc_specred_out/1501662.1.N3_NP.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/1501662.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
1600060.1/irc_specred_out/1600060.1.N3_NG.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/1600060.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
1720004.1/irc_specred_out/1720004.1.N3_NG.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/1720004.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
1720005.1/irc_specred_out/1720005.1.N3_NG.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/1720005.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
3261008.1/irc_specred_out/3261008.1.N3_NP.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/3261008.1/reg2


python ext_spec.py \
/home/morii/work/akari/data/spectrum/DATA/\
3261087.1/irc_specred_out/3261087.1.N3_NP.specimage_bg.fits \
$reg2_corner \
/home/morii/work/akari/ana/spectrum/3261087.1/reg2

