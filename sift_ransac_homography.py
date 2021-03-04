import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import torch
import PIL.Image as image
from torchvision import models, transforms

from matplotlib import pyplot as plt

from torch.autograd import Variable
import exifread


# sift = cv2.xfeatures2d.SIFT_create()

# kp1, des1 = sift.detectAndCompute(gray,None)
# kp2, des2 = sift.detectAndCompute(dst_gray,None)

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)

# # store all the good matches as per Lowe's ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
# M_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# # gray_kp = cv2.drawKeypoints(gray,kp1,None)
# # gray_dst_kp = cv2.drawKeypoints(dst_gray,kp2,None)

# # plt.subplot(121),plt.imshow(gray_kp),plt.title('Input')
# # plt.subplot(122),plt.imshow(gray_dst_kp),plt.title('Output')
# # plt.show()

# # set_trace()

# # kp1_loc = np.float32([kp1[i].pt for i in range(len(kp1))])
# # kp2_loc = np.float32([kp2[i].pt for i in range(len(kp2))])

# # M_found, mask = cv2.findHomography(kp1_loc, kp2_loc, cv2.RANSAC, 5.0)

# print(H_gt)
# print(M_found)

# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_param(img_batch, template_batch, image_sz):
	template = template_batch.data.squeeze(0).cpu().numpy()
	img = img_batch.data.squeeze(0).cpu().numpy()

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
	# sift = cv2.SIFT()

	kp1, des1 = sift.detectAndCompute(template_gray, None)
	kp2, des2 = sift.detectAndCompute(img_gray, None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	img3 = cv2.drawMatches(template_gray, kp1, img_gray, kp2, matches[:20], None, flags=2)
	plt.imshow(img3), plt.show()

	set_trace()

	# template_gray_with_kp = cv2.drawKeypoints(template_gray,kp1,None)
	# img_gray_with_kp = cv2.drawKeypoints(img_gray,kp2,None)
	# cv2.imshow('template',template_gray_with_kp)
	# cv2.imshow('image',img_gray_with_kp)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

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

	H = torch.from_numpy(H_found).float()
	I = torch.eye(3, 3)

	p = H - I

	p = p.view(1, 9, 1)
	p = p[:, 0:8, :]

	if torch.cuda.is_available():
		return Variable(p.cuda())
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
	M_with_marker_np[x_cor - 25:x_cor + 25, y_cor - 25:y_cor + 25, :] = color_paint
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
	fd = open(img_name, 'rb')
	tags = exifread.process_file(fd)
	fd.close()

	lag = tags.get("GPS GPSLatitude")
	lag_ref = tags.get("GPS GPSLatitudeRef")
	log = tags.get("GPS GPSLongitude")
	log_ref = tags.get("GPS GPSLongitudeRef")

	lag = lag.values
	log = log.values
	lag_numberic = float(lag[0]) + float(lag[1] / 60) + float(lag[2]) / 3600
	log_numberic = float(log[0]) + float(log[1] / 60) + float(log[2]) / 3600

	return [lag_numberic, lag_ref, log_numberic, log_ref]


def lag_log_to_pix_pos(M, GPS_target):
	GPS_1 = [47.0705666666667, 8.42165833333333]
	GPS_2 = [47.0591972222222, 8.39368611111111]
	if type(M) == np.ndarray:
		c, h, w = M.shape
	else:
		w, h = M.size()
	target_lag = GPS_target[0]
	target_log = GPS_target[1]
	Map_lag_1 = GPS_1[0]
	Map_lag_2 = GPS_2[0]
	Map_log_1 = GPS_1[1]
	Map_log_2 = GPS_2[1]
	target_pix_pos_x = h * ((target_lag - Map_lag_1) / (Map_lag_2 - Map_lag_1))
	target_pix_pos_y = w * ((target_log - Map_log_2) / (Map_log_2 - Map_log_1))
	return [int(abs(target_pix_pos_x)), int(abs(target_pix_pos_y))]


if __name__ == "__main__":
	img = cv2.imread('../duck.jpg')
	img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	rows, cols, ch = img_color.shape

	pts1 = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]])
	pts2 = np.float32([[0, 0], [0, rows], [cols + 200, rows], [cols + 200, 0]])

	H_gt = cv2.getPerspectiveTransform(pts1, pts2)

	print(H_gt)

	dst_img = cv2.warpPerspective(img_color, H_gt, (cols, rows))

	template_batch = Variable(torch.from_numpy(img_color).unsqueeze(0))
	img_batch = Variable(torch.from_numpy(dst_img).unsqueeze(0))

	print(get_param(img_batch, template_batch))