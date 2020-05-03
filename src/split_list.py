
import numpy as np

txt_file = '../output/kitti_train.txt'
val_range = 200

with open(txt_file, 'r') as f:
    train_file = open('fo_kitti_train.txt', 'w')
    valid_file = open('fo_kitti_val.txt', 'w')
    lines = f.readlines()
    val_inds = np.random.randint(0, len(lines)-1, 200) 
    for ind, line in enumerate(lines):
        if ind in val_inds:
            valid_file.write(line)
        else:
            train_file.write(line)
    train_file.close()
    valid_file.close()
