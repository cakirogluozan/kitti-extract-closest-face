import os
import numpy as np
from utils import write_labels

if __name__ == '__main__':

    # TODO:
    CALIB_DIR = "TODO"
    IMAGE_DIR = "TODO"
    LABEL_DIR = "TODO"

    calib_list = [os.path.join(CALIB_DIR, png) for png in os.listdir(CALIB_DIR) if png.endswith('.txt')]
    calib_list.sort()
    image_list = [os.path.join(IMAGE_DIR, png) for png in os.listdir(IMAGE_DIR) if png.endswith('.png')]
    image_list.sort()
    label_list = [os.path.join(LABEL_DIR, png) for png in os.listdir(LABEL_DIR) if png.endswith('.txt')]
    label_list.sort()
    
    write_labels(image_list, label_list, calib_list)