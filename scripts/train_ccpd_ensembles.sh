#!/bin/bash

python train.py --data data/ccpd.data --epochs 100 --batch-size 16 --name ccpd_ensemble_1 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 416
python train.py --data data/ccpd.data --epochs 100 --batch-size 16 --name ccpd_ensemble_2 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 416
python train.py --data data/ccpd.data --epochs 100 --batch-size 16 --name ccpd_ensemble_3 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 416
python train.py --data data/ccpd.data --epochs 100 --batch-size 16 --name ccpd_ensemble_4 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 416
python train.py --data data/ccpd.data --epochs 100 --batch-size 16 --name ccpd_ensemble_5 --weights '' --cfg cfg/yolov3-custom-ccpd.cfg --img-size 416