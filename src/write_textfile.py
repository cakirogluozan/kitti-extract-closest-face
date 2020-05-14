import os
import numpy as np
from utils import write_labels

if __name__ == '__main__':

    # TODO:
    dgxs=False

    KITTI_DIR = "/home/ocakirog/Desktop/Datasets/KITTI/"
    CALIB_DIR = os.path.join(KITTI_DIR, 'data_object_calib/training/calib')
    IMAGE_DIR = os.path.join(KITTI_DIR, 'data_object_image_2/training/image_2')
    LABEL_DIR = os.path.join(KITTI_DIR, 'data_object_label_2/training/label_2')

    if dgxs:
        CALIB_DIR = CALIB_DIR.replace('/raid', '/mnt/dgx1_data')
        IMAGE_DIR = IMAGE_DIR.replace('/raid', '/mnt/dgx1_data')
        LABEL_DIR = LABEL_DIR.replace('/raid', '/mnt/dgx1_data')
    ######


    calib_list = [os.path.join(CALIB_DIR, png) for png in os.listdir(CALIB_DIR) if png.endswith('.txt')]
    calib_list.sort()
    image_list = [os.path.join(IMAGE_DIR, png) for png in os.listdir(IMAGE_DIR) if png.endswith('.png')]
    image_list.sort()
    label_list = [os.path.join(LABEL_DIR, png) for png in os.listdir(LABEL_DIR) if png.endswith('.txt')]
    label_list.sort()
    
    write_labels(image_list, label_list, calib_list)