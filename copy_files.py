
import os
from pathlib import Path
from shutil import copyfile
import sys
import cv2

from torch import tensor
from torchvision.ops import box_convert


FIELDS = [
    "area",
    "tilt_degree",
    "bounding_box",
    "vertices",
    "license_plate",
    "brightness",
    "blurriness",
]


working_directory = Path(os.getcwd())
splits = working_directory / '..' /'splits'

def load_file(path):
    with open(path) as f:
        files = [l.strip() for l in f.readlines()]
    return files

def copy_image(file, folder):
    filename = file.split('/')[-1]
    current_location = working_directory / '..' / file
    new_location = working_directory / 'images' / folder / filename
    copyfile(src=current_location, dst=new_location)

def parse_filename(file_path: str):
    values = file_path.split("/")[-1].rsplit(".", 1)[0].split("-")
    value_dict = dict(zip(FIELDS, values))
    return value_dict

def create_label_txt(file, width, height, folder):
    values = parse_filename(file)
    bbox_points = values['bounding_box'].split('_')
    bbox = [int(coord) for coord in bbox_points[0].split('&') + bbox_points[1].split('&')]
    converted_bbox = box_convert(tensor(bbox), in_fmt='xyxy', out_fmt='xywh')
    new_converted_box = converted_bbox.tolist()
    new_converted_box[0] = new_converted_box[0]/width
    new_converted_box[1] = new_converted_box[1]/height
    new_converted_box[2] = new_converted_box[2]/width
    new_converted_box[3] = new_converted_box[3]/height
    box_to_strings = ["{:.6f}".format(v) for v in new_converted_box]
    box_as_string = ' '.join(box_to_strings)
    filename = file.split('/')[-1].rsplit(".",1)[0]
    save_location = working_directory / 'labels' / folder / f'{filename}.txt'
    with open(save_location, mode='w') as f:
        f.write(f'0 {box_as_string}')


def create_file_txt(files, file_type):
    file_name = f'{file_type}.txt'
    file_path = working_directory / file_name
    elements = [f'./images/{file_type}/{f.split("/")[-1]}\n' for f in files]
    with open(file_path, mode='w') as f:
        f.writelines(elements)


if __name__ == '__main__':
        
    train_files = load_file(splits/'train.txt')
    val_files = load_file(splits/'val.txt')

    for f in train_files[0:20]:
        copy_image(f, 'train_mini')
        image = cv2.imread(f'../{f}')
        width = image.shape[1]
        height = image.shape[0]
        create_label_txt(f,width, height , 'train_mini')

    for f in val_files[0:20]:
        copy_image(f, 'val_mini')
        image = cv2.imread(f'../{f}')
        width = image.shape[1]
        height = image.shape[0]
        create_label_txt(f,width, height , 'val_mini')

    create_file_txt(train_files[0:20], 'train_mini')
    create_file_txt(val_files[0:20], 'val_mini')
