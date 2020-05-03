import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d


# =========================================================
# Projections
# =========================================================

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def map_box_to_image(box, proj_mat):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D)
        proj_mat: projection matrix
    """
    # box in camera coordinate
    points_3d = box.in_camera_coordinate()

    # project the 3d bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat)

    return points_2d

def get_valid_kitti_face(objects, calib, image_shape, yaw_th):
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    class_list  = list()
    bbox_list  = list()
   
    for obj in objects:
        if obj.type == 'DontCare' or abs(obj.ry) < yaw_th or obj.type not in ['Car', 'Van', 'Truck', 'Bus']:
            continue
            
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        if obj.ry > yaw_th:
            valid_pts_ind = [0, 1, 4, 5] # the vehicle's face
            class_ = 'front'
        else :
            valid_pts_ind = [2, 3, 6, 7]  # the vehichle' ass
            class_ = 'back'
            
        pts_list    = list()
        # print(image_shape, 'image shape')
        flag_val, flag_int = False, False
        for ind in range(len(box3d_pixelcoord[0])):
            if ind in valid_pts_ind:
                flag_val = True
                x = int(box3d_pixelcoord[0][ind])
                y = int(box3d_pixelcoord[1][ind])
                if x < 0 or y < 0 or x >= image_shape[1] or y >= image_shape[0]:
                    flag_int = True
                pts_list.append((x, y)) 
        if not flag_int and flag_val:
            bbox_list.append(pts_list)
            class_list.append(class_)
    return bbox_list, class_list


def visualize_bbox(image, bbox_list, class_list):
    for ind in range(len(bbox_list)):
        bbox   = bbox_list[ind]
        class_ = class_list[ind]
        (x1, y1) = bbox[3]
        (x2, y2) = bbox[0]
        
        if class_ == 'front':
            color    = (0, 255, 255)    
        elif class_ == 'back':
            color    = (255, 0, 255)
        else:
            break
            
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, class_, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
    plt.figure(figsize=(25,20));
    plt.imshow(image);

def label_line(image_path, bbox_list, class_list):
    # image_path xmin,ymin,xmax,ymax,classid xmin..    
    if len(bbox_list) == 0:
        return None
    line = "{} ".format(image_path)

    for ind in range(len(bbox_list)):
        bbox   = bbox_list[ind]
        (x1, y1) = bbox[3]
        (x2, y2) = bbox[0]
        class_ = class_list[ind]
        class_id = 1 if class_ == 'front' else 0
        obj_line = "{},{},{},{},{}".format(x1,y1,x2,y2,class_id)
        if ind == len(bbox_list)-1:
            obj_line += "\n"
        else:
            obj_line += " "
        line += obj_line
    return line

def write_labels(image_list, label_list, calib_list, visualize=False, yaw_th=0.4):
    with open('../output/kitti_train.txt', 'w') as f:
        for i in trange(len(label_list)):
            calib  = read_calib_file(calib_list[i])
            objects = load_label(label_list[i])
            image  = plt.imread(image_list[i])

            bbox_list, class_list = get_valid_kitti_face(objects, calib, image.shape, yaw_th)
            line = label_line(image_list[i], bbox_list, class_list)
            if type(line) != type(None):
                f.write(line)
            if visualize:
                visualize_bbox(image, bbox_list, class_list)       

    f.close()
# =========================================================
# Utils
# =========================================================
def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


# =========================================================
# Drawing tool
# =========================================================


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    qs = qs.astype(np.int32).transpose()
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    return image

