# 测试的数据集
test_dataset = "Gravel_Pit"

# general parameters
tol = 1e-3
max_itr = 10
lam1 = 1
lam2 = 0.01
scaling_for_disp = 1
max_itr_dlk = 150

if test_dataset == "haidian_airport":
    image_dir = "./haidian_airport/camera1/"  # TODO: the directory of the captured frames
    image_dir_ext = "*.png"
    satellite_map_name = 'haidian_airport'
    heading_dct = 'east'
    map_width = 4870
    map_height = 2910
    map_loc = "./haidian_airport/map_airport.tif"
    model_path = "./models/best_dlk_model_res18_new.pth"
    opt_param_save_loc = "../haidian_airport/test_out.mat"
    image_idx = 30
    interval = 5

    img_height_opt_net = 100
    img_height_rel_pose = 600
    map_resolution = 0.465
    scale_img_to_map = map_height / img_height_rel_pose

    init_x = 3780  # TODO: initial x coordinate in image plane
    init_y = 1800  # TODO: initial y coordinate in image plane
    init_scale = 1 / 6  # TODO: initial scale
    init_angle = 0  # TODO: initial angle

    Map_GPS_1 = [116.116820683836, 40.0856287594884]
    Map_GPS_2 = [116.10104906676413, 40.0652127940419]


elif test_dataset == "village":
    # Village Dataset
    image_dir = "./village/frames_with_geotag_1/"
    image_dir_ext = "*.JPG"
    satellite_map_name = 'village'
    map_loc = "./village/map_village.jpg"
    heading_dct = 'north'
    map_width = 4800
    map_height = 2861

    model_path = "./models/best_dlk_model_raw.pth"
    img_height_opt_net = 100
    img_height_rel_pose = 600
    opt_param_save_loc = "./village/test_out.mat"
    map_resolution = 0.45
    scale_img_to_map = map_height / img_height_rel_pose
    image_idx = 0
    interval = 1

    init_x = 2837
    init_y = 841
    init_scale = 1 / 6
    init_angle = - 135

    Map_GPS_1 = [8.39368611111111, 47.0705666666667]
    Map_GPS_2 = [8.42165833333333, 47.0591972222222]

elif test_dataset == "Gravel_Pit":
    # Gravel-Pit Dataset
    image_dir = "./gravel_pit/frames_with_geotag/"
    image_dir_ext = "*.JPG"
    satellite_map_name = 'gravel-pit'
    map_loc = "./gravel_pit/map_gravel_pit.jpg"
    heading_dct = 'north'
    map_width = 3355
    map_height = 1852

    model_path = "./models/best_dlk_model_res18_new.pth"
    img_height_opt_net = 100
    img_height_rel_pose = 800
    opt_param_save_loc = "../gravel_pit/test_out.mat"
    map_resolution = 0.32
    scale_img_to_map = map_height / img_height_rel_pose
    image_idx = 0
    interval = 1

    init_x = 2500
    init_y = 1050
    init_scale = 1 / 4.3
    init_angle = -70

    Map_GPS_1 = [7.90523333333333, 47.1148611111111]
    Map_GPS_2 = [7.91988055555556, 47.1093700000000]

else:
    raise NameError('Can\'t the dataset')
