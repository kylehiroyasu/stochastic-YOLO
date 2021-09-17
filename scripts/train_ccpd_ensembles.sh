#!/bin/bash

python train.py --data data/ccpd.data --epochs 50 --batch-size 16 --name ccpd_ensemble_1 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 640 --notest
python train.py --data data/ccpd.data --epochs 50 --batch-size 16 --name ccpd_ensemble_2 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 640 --notest
python train.py --data data/ccpd.data --epochs 50 --batch-size 16 --name ccpd_ensemble_3 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 640 --notest
python train.py --data data/ccpd.data --epochs 50 --batch-size 16 --name ccpd_mcdrop0 --weights '' --cfg cfg/yolov3-mcdrop0-ccpd.cfg --img-size 640 --notest
