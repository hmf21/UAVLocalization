import os
import sys
import warnings
import autograd.numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import transforms
from utility import config
import glob
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, './deep_feat')  # for model loading

# control the result visualize and record
show_fig = True
show_error = True
record_data = True * show_error


def UAV_loc(Init_UAV_info):
    return 0


def main():
    # setting parameters for showing the result
    warnings.filterwarnings("ignore")
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize = (10, 5))
    np.set_printoptions(precision = 4)

    # parameters setting of localization
    # we can set the initial position of UAV if it is available
    Init_UAV_info = {
        'image_list': None,
        'trans_x': config.init_x,
        'trans_y': config.init_y,
        'angle': config.init_angle,
        'scale': config.init_scale,
        'altitude': None,
        'd_x': 0,
        'd_y': 0,
        'd_angle': 0,
        'd_scale': 0,
        'curr_image': None,
        'prev_image': None,
        'curr_image_for_rel_loc': None,
        'prev_image_for_rel_loc': None,
        'curr_image_for_match_loc': None,
        'prev_image_for_match_loc': None,
        'extract_map_patch': None,
        'curr_GPS': None,
        'is_located': False,
        'image_idx': 0,
    }
    print("Testing image directory is : ", os.path.join(config.image_dir, config.image_dir_ext))
    image_list = sorted(glob.glob(os.path.join(config.image_dir, config.image_dir_ext)))
    Init_UAV_info['image_idx'] = config.image_idx
    Init_UAV_info['curr_image'] = image_list[Init_UAV_info['image_idx']]

    # start localization process
    Curr_UAV_info = Init_UAV_info
    while True:
        start_ts = time.time()
        Curr_UAV_info = UAV_loc(Curr_UAV_info)
        print("totally spent time: ", time.time() - start_ts)
    return 0


if __name__ == "__main__":
    xy_cor_list = []
    xy_gps_list_opt = []
    xy_cor_list_opt = []
    xy_cor_list_gt = []
    loc_img_list = []
    main()
