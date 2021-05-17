import cv2
import numpy as np
import torch
import PIL.Image as image
from torchvision import models, transforms
import glob
from PIL import Image
import math

from torch.autograd import Variable
import exifread
import img_load as img_utility
from config import image_dir
from config import motion_param_loc
from config import opt_img_height
from config import GPS_1 as Map_GPS_1
from config import GPS_2 as Map_GPS_2


def get_param(img_batch, template_batch, image_sz):
	template = template_batch.data.squeeze(0).cpu().numpy()
	img = img_batch.data.squeeze(0).cpu().numpy()
	sc = image_sz / img.shape[1]

	if template.shape[0] == 3:
		template = np.swapaxes(template, 0, 2)
		template = np.swapaxes(template, 0, 1)
		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 0, 1)

		template = (template * 255).astype('uint8')
		img = (img * 255).astype('uint8')

	# set_trace()

	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	# sift = cv2.xfeatures2d.SURF_create()
	# sift = cv2.SIFT()
	# template 在第一个参数位置，img在第二个参数，这里的计算方式是与
	kp1, des1 = sift.detectAndCompute(template_gray, None)
	kp2, des2 = sift.detectAndCompute(img_gray, None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	img3 = cv2.drawMatches(template_gray, kp1, img_gray, kp2, matches[:20], None, flags=2)
	# plt.imshow(img3)
	# plt.show()

	# set_trace()

	# template_gray_with_kp = cv2.drawKeypoints(template_gray,kp1,None)
	# img_gray_with_kp = cv2.drawKeypoints(img_gray,kp2,None)
	# cv2.imshow('template',template_gray_with_kp)
	# cv2.imshow('image',img_gray_with_kp)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# plt.imshow(template_gray)
	# plt.imshow(img_gray_with_kp)
	# plt.show()

	if (len(kp1) >= 2) and (len(kp2) >= 2):

		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)

		# store all the good matches as per Lowe's ratio test
		good = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good.append(m)

		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		# 这里的点对不做归一化可否？
		src_pts = src_pts - image_sz / 2
		dst_pts = dst_pts - image_sz / 2

		if (src_pts.size == 0) or (dst_pts.size == 0):
			H_found = np.eye(3)
		else:
			H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

		if H_found is None:
			H_found = np.eye(3)

	else:
		H_found = np.eye(3)

	Perspective_img = cv2.warpPerspective(template, H_found, (template.shape[1], template.shape[0]))
	# plt.imshow(Perspective_img)
	# plt.show()

	H = torch.from_numpy(H_found).float()
	I = torch.eye(3, 3)

	p = H - I

	p = p.view(1, 9, 1)
	p = p[:, 0:8, :]
	# p = p * 3

	if torch.cuda.is_available():
		# return Variable(p.cuda())
		return Variable(p)
	else:
		return Variable(p)


# 增加画图的函数
def Add_M_Marker(M, xy_cor, color="red"):
	if color == "red":
		color_paint = [255, 0, 0]
	elif color == "blue":
		color_paint = [0, 0, 255]
	elif color == "green":
		color_paint = [0, 255, 0]
	else:
		color_paint = [255, 0, 0]
	w, h = M.size
	x_cor = round(xy_cor[1] / 1)
	y_cor = round(xy_cor[0] * 1)
	M_with_marker_np = np.copy(M)
	M_with_marker_np[x_cor - 10:x_cor + 10, y_cor - 10:y_cor + 10, :] = color_paint
	M_with_marker_pil = image.fromarray(M_with_marker_np)
	return M_with_marker_pil


def Add_M_Line(M, xy_cor_1, xy_cor_2):
	# ax+by=c
	# 感觉会比较难写啊
	w, h = M.size
	x_1 = xy_cor_1[0]
	y_1 = xy_cor_1[1]
	x_2 = xy_cor_2[0]
	y_2 = xy_cor_2[1]
	k = (y_1 - y_2) / (x_1 - x_2)
	b = (x_1 * y_2 - x_2 * y_1) / (x_1 - x_2)
	for i in range(h):
		for j in range(w):
			if abs(k * i + b - j) < 10:
				M[i, j, :] = [255, 0, 0]
	return M


def Add_M_Markers_list(M, xy_cor_list, color="red"):
	# 循环添加
	if type(M) is np.ndarray:
		M_pil = transforms.ToPILImage()(torch.from_numpy(M))
	else:
		M_pil = M
	xy_cor_cur = []
	xy_cor_pre = []
	for xy_cor in xy_cor_list:
		xy_cor_cur = xy_cor_pre
		M_pil = Add_M_Marker(M_pil, xy_cor, color=color)
		if xy_cor_pre != []:
			M_pil = Add_M_Line(M_pil, xy_cor_pre, xy_cor_cur)
	return M_pil


def get_img_log_lat(img_name):
	""""
	获得标准图片中所含有的GPS信息，适用于文章中使用的village数据集以及gravel_pit数据集
	"""
	fd = open(img_name, 'rb')
	tags = exifread.process_file(fd)
	fd.close()

	lag = tags.get("GPS GPSLatitude")
	lag_ref = tags.get("GPS GPSLatitudeRef")
	log = tags.get("GPS GPSLongitude")
	log_ref = tags.get("GPS GPSLongitudeRef")

	if lag == None or log == None:
		lag_numberic = None
		log_numberic = None
	else:
		lag = lag.values
		log = log.values
		lag_numberic = float(lag[0]) + float(lag[1] / 60) + float(lag[2]) / 3600
		log_numberic = float(log[0]) + float(log[1] / 60) + float(log[2]) / 3600

	return [lag_numberic, lag_ref, log_numberic, log_ref]


def lag_log_to_pix_pos(M, GPS_target):
	# for village dataset
	if math.isnan(GPS_target[0]) or math.isnan(GPS_target[1]):
		return [0, 0]
	if type(M) == np.ndarray:
		c, h, w = M.shape
	else:
		w, h = M.size()
	target_lag = GPS_target[0]
	target_log = GPS_target[1]
	Map_lag_1 = Map_GPS_1[0]
	Map_lag_2 = Map_GPS_2[0]
	Map_log_1 = Map_GPS_1[1]
	Map_log_2 = Map_GPS_2[1]
	target_pix_pos_x = h * ((target_lag - Map_lag_1) / (Map_lag_2 - Map_lag_1))
	target_pix_pos_y = w * ((target_log - Map_log_2) / (Map_log_2 - Map_log_1))
	return [int(abs(target_pix_pos_y)), int(abs(target_pix_pos_x))]


def cal_P_init(img_sz):
	# M_cols = 2000
	# M_rows = 1330
	# I_cols = 2000q
	# I_rows = 1330
	# 利用真实地理位置计算P_init
	M_GPS_1 = [37.9593808178314, -122.496427618767]
	M_GPS_2 = [37.9235161004564, -122.542184143707]
	M_height = 8000
	M_width = 8000
	D_GPS_1 = [37.9482770280025, -122.513931274564]
	D_GPS_2 = [37.9259558386743, -122.5341316126]
	I_height = 1330
	I_width = 2000
	D_height = 16000
	D_width = 12000
	x_coor_init = 1467
	y_coor_init = 5949
	# 八点对
	I_init_in_Drone_pts = [[int(x_coor_init-I_height/2),int(y_coor_init-I_width/2)],[int(x_coor_init-I_height/2),y_coor_init],
					 [int(x_coor_init-I_height/2),int(y_coor_init+I_width/2)],[x_coor_init,int(y_coor_init-I_width/2)],
					 [x_coor_init,int(y_coor_init+I_width/2)],[int(x_coor_init+I_height/2),int(y_coor_init-I_width/2)],
					 [int(x_coor_init+I_height/2),y_coor_init],[int(x_coor_init+I_height/2),int(y_coor_init+I_width/2)]]
	I_init_in_Map_pts = []
	for I_init_in_Drone_pt in I_init_in_Drone_pts:
		I_GPS_lag = D_GPS_1[0] - (D_GPS_1[0]-D_GPS_2[0])*I_init_in_Drone_pt[0]/D_height
		I_GPS_log = (D_GPS_1[1]-D_GPS_2[1])*I_init_in_Drone_pt[1]/D_width+D_GPS_2[1]
		I_init_in_Map_x_coor = int((M_GPS_1[0]-I_GPS_lag)/(M_GPS_1[0]-M_GPS_2[0])*M_height)
		I_init_in_Map_y_coor = int((I_GPS_log-M_GPS_2[1])/(M_GPS_1[1]-M_GPS_2[1])*M_width)
		I_init_in_Map_pts.append([I_init_in_Map_x_coor,I_init_in_Map_y_coor])
	# 计算出点对
	src_pts = [[670*2,0],[670*2,4000],
			   [670*2,2000*4],[670*2+665*4,0],
			   [670*2+665*4,2000*4],[670*2+1330*4,0],
			   [670*2+1330*4,1000*4],[670*2+1330*4,2000*4]]


	src_pts = [[1523,2400],[1523,2401],[1524,2400],
			   [1522,2400],[1523,2399],[1524,2401],
			   [1524,2399],[1522,2401],[1522,2399]]
	I_init_in_Map_pts = [[2802,243],[2802,244],[2803,243],
						 [2801,243],[2802,242],[2803,244],
						 [2803,242],[2801,244],[2801,242]]
	intervel = 50
	intervel_M = 100
	I_init_in_Map_pts = [[0,0],[0,2400],[0,4800],[1524,0],[1524,4800],[3048,0],[3048,2400],[3048,4800]]
	src_pts = [[0,0],[0,600],[0,1200],[381,0],[381,1200],[762,0],[762,600],[762,1200]]
	src_pts = src_pts + np.array([-1900, 1100])

	I_init_in_Map_pts = [[0,0],[0,4600],[2748,0],[2748,4600]]
	src_pts = [[0,0],[0,1150],[687,0],[687,1150]]
	src_pts = np.array(src_pts)
	I_init_in_Map_pts = np.array(I_init_in_Map_pts)

	# I_init_in_Map_pts = [[0, 400], [0, 4400], [2600, 400], [2600, 4400]]
	# src_pts = [[1903, 1089], [2467, 1539], [1912, 520], [2451, 790]]
	src_pts = np.array(src_pts)
	I_init_in_Map_pts = np.array(I_init_in_Map_pts)

	# src_pts = src_pts - img_sz / 2
	# I_init_in_Map_pts = I_init_in_Map_pts - img_sz / 2
	src_pts = src_pts * 0.6
	I_init_in_Map_pts = I_init_in_Map_pts * 0.6
	H_found, mask = cv2.findHomography(src_pts, I_init_in_Map_pts, cv2.RANSAC, 5.0)
	H = torch.from_numpy(H_found).float()
	I = torch.eye(3, 3)
	p = H - I
	p = p.view(1, 9, 1)
	p = p[:, 0:8, :]
	return p


def main():
	# 自模拟数据集的保存
	# scaled_im_height是单应矩阵计算的关键，这里当采样越大的时候，计算越精确，因为很大一部分会与采样有关
	# 但是我们设计的网络按理说也应该有这样的一个性质，所以也是需要注意的问题
	I_list, _ = img_utility.load_I(image_dir, ext="*.jpg", scaled_im_height=opt_img_height)
	p = []
	P_init = cal_P_init(img_sz=4800)
	p.append(np.array(P_init))
	# p.append(np.zeros([1, 8, 1]))
	for i in range(I_list.shape[0]-1):
		I_1 = I_list[i:i + 1, :, :, :]
		I_2 = I_list[i + 1:i + 2, :, :, :]
		template_batch = Variable(torch.from_numpy(I_1).squeeze(0))
		img_batch = Variable(torch.from_numpy(I_2).squeeze(0))
		tmp = get_param(img_batch, template_batch, opt_img_height)
		p.append(np.array(tmp))
	np.savetxt(motion_param_loc, np.squeeze(np.array(p)), delimiter=',')


if __name__ == "__main__":
	# p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
	# 							  -sin(rad_ang),
	# 							  translation_x,
	# 							  sin(rad_ang),
	# 							  scale + cos(rad_ang) - 2,
	# 							  translation_y,
	# 							  projective_x,
	# 							  projective_y]))
	main()
	pass