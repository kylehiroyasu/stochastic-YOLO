FROM syolo-base
WORKDIR /project
COPY . .
CMD ["conda", "run" "-n", "alpr", "python", "train.py", "--data", "data/ccpd_min.data", "--epochs", "1", "--batch-size", "16", "--name", "coco2017_scratch", "--weights", "", "--cfg", "cfg/yolov3-custom-ccpd.cfg", "--img-size", "416"]