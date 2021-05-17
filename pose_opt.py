import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
from torch.autograd import Variable
from math import cos, sin, pi, sqrt, ceil
import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy.io as sio
import warnings

import DeepLKBatch as dlk
import sift_ransac_homography as srh
import img_load as img_utility
import pama_load as pama_utility
from config import *
import time


def optimize(I, P, T, V, tol, coeff_mult):
	'''
	:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	Returns:
		P_opt: Bundle adjusted warp parameter sequence, 3D numpy array, num_frame x 8 x 1
	'''

	num_frame = I.shape[0]

	def size_fn(a):
		return a.size

	size_fn_vec = np.vectorize(size_fn)

	V_sz = size_fn_vec(V)
	sigma = V_sz.sum()

	_, img_c, img_h, img_w = I.shape

	I_gradx = np.gradient(I, axis=3)
	I_grady = np.gradient(I, axis=2)

	dP = np.zeros([num_frame - 1, 8, 1])
	crit = 0
	itn = 1
	coeff = 1

	while ((crit > tol) or (itn == 1)) and (itn < 200):
		P_fk = compute_Pfk(P, T, V, V_sz)

		r_ba = compute_ri(I, T, V, P_fk, V_sz)

		J_ba = compute_Ji(I_gradx, I_grady, P, T, V, P_fk, V_sz)

		J_ba_trans = J_ba.swapaxes(2, 3)

		J_ba_trans_J_ba = np.matmul(J_ba_trans, J_ba)

		Hess = np.sum(J_ba_trans_J_ba, axis=1)

		invHess = np.linalg.inv(Hess)

		J_ba_trans_r_ba = np.matmul(J_ba_trans, r_ba)

		J_ba_trans_r_ba_sum = np.sum(J_ba_trans_r_ba, axis=1)

		dP = np.matmul(invHess, J_ba_trans_r_ba_sum)

		P = P + dP * coeff

		dp_norm = np.linalg.norm(dP, ord=2, axis=1)
		crit = np.amax(dp_norm)

		if (itn % 10 == 0):
			coeff = coeff * coeff_mult

		print('itn:  {:d}'.format(itn))
		print('crit: {:.5f}'.format(crit))

		# print('coeff: {:.2f}'.format(coeff))
		# print(P)
		# print(dP)

		itn = itn + 1

	return P


def optimize_wmap(I, P, T, V, P_init, M_feat, T_feat, tol, max_itr, lam1, lam2):
	'''
	:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		M: Map image sequence, 4D numpy array, k x C x H x W
		T: Numpy vector containing indices from I which are templates, length k
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_init: Absolute warp parameters relating map to first frame, 1 x 8 x 1
		M_feat: Sequence of deep feature map images, 4D numpy array, k x Cf x Hf x Wf
		T_feat: Template images from I deep feature extractions, 4D numpy array, k x Cf x Hf x Wf
		tol: tolerance of iteration criteria
		max_itr: maximum number of optimization iterations
		lam1: weighting for sequence motion parameters
		lam2: weight for map params

	Returns:
		P_opt: Bundle adjusted warp parameter sequence, 3D numpy array, num_frame x 8 x 1
	'''

	num_frames = I.shape[0]

	def size_fn(a):
		if type(a) is int:
			return 1
		else:
			return a.size

	size_fn_vec = np.vectorize(size_fn)

	V_sz = size_fn_vec(V)
	sigma = V_sz.sum()

	I_gradx = np.gradient(I, axis=3)
	I_grady = np.gradient(I, axis=2)

	Mf_gradx = np.gradient(M_feat, axis=3)
	Mf_grady = np.gradient(M_feat, axis=2)

	P_mk0 = compute_Pmk(P_init, P, T)

	dP = np.zeros([num_frames - 1, 8, 1])
	crit = 0

	itn = 1

	while ((crit > tol) or (itn == 1)) and (itn < max_itr):
		P_fk = compute_Pfk(P, T, V, V_sz)
		P_mk = compute_Pmk(P_init, P, T)

		# warping: H_P * inv(H_P0) <- by rules of homography

		# convert sampling params to sampling hmg:
		H_mk_samp = pama_utility.p_to_H(P_mk)
		H_mk0_samp = pama_utility.p_to_H(P_mk0)

		# invert sampling params to get coord params:
		H_mk = np.linalg.inv(H_mk_samp)
		H_mk0 = np.linalg.inv(H_mk0_samp)

		# compute relative hmg:
		H_mk_rel = np.matmul(H_mk, np.linalg.inv(H_mk0))

		# invert back to sampling hmg:
		H_mk_rel_samp = np.linalg.inv(H_mk_rel)

		# convert sampling hmg back to sampling params:
		P_mk_rel_samp = pama_utility.H_to_p(H_mk_rel_samp)

		ri = compute_ri(I, T, V, P_fk, V_sz)

		rm_rsh = compute_rm(M_feat, T_feat, P_mk_rel_samp)
		rm = np.tile(rm_rsh, (num_frames - 1, 1, 1, 1))

		Ji = compute_Ji(I_gradx, I_grady, P, T, V, P_fk, V_sz)
		Jm = compute_Jm(Mf_gradx, Mf_grady, P_init, P, T, P_mk_rel_samp, P_mk0)

		Ji_trans = Ji.swapaxes(2, 3)
		Ji_trans_Ji = lam1 * np.matmul(Ji_trans, Ji)

		Jm_trans = Jm.swapaxes(2, 3)
		Jm_trans_Jm = lam2 * np.matmul(Jm_trans, Jm)

		Hm = np.sum(Jm_trans_Jm, axis=1)
		Hi = np.sum(Ji_trans_Ji, axis=1)

		invH = np.linalg.inv(Hm + Hi)

		Jm_trans_rm = lam2 * np.matmul(Jm_trans, rm)
		Ji_trans_ri = lam1 * np.matmul(Ji_trans, ri)

		Jm_trans_rm_sum = np.sum(Jm_trans_rm, axis=1)
		Ji_trans_ri_sum = np.sum(Ji_trans_ri, axis=1)

		dP = np.matmul(invH, Jm_trans_rm_sum + Ji_trans_ri_sum)

		dP[:, 6:8, :] = 0

		P = P + dP

		dp_norm = np.linalg.norm(dP, ord=2, axis=1)
		crit = np.amax(dp_norm)

		print('itn:  {:d}'.format(itn))
		print('crit: {:.5f}'.format(crit))

		itn = itn + 1

	return P


def compute_rm(M_feat, T_feat, P_mk_rel):
	'''
	:
		M_feat: Sequence of deep feature map images, 4D numpy array, k x Cf x Hf x Wf
		T_feat: Template images from I deep feature extractions, 4D numpy array, k x Cf x Hf x Wf
		P_mk_rel: Numpy array, warp parameters for each of the map images, k x 8 x 1

	Returns:
		rm: Numpy array, residuals of map images with templates, num_frame - 1 (duplicated dim.) x k x (Cf x Hf x Wf) x 1
	'''

	k, map_c, map_h, map_w = M_feat.shape

	P_mk_rel_tens = torch.from_numpy(P_mk_rel).float()
	M_feat_tens = torch.from_numpy(M_feat).float()

	M_feat_warp_tens, M_feat_mask_tens, _ = dlk.warp_hmg(M_feat_tens, P_mk_rel_tens)

	M_feat_warp = M_feat_warp_tens.numpy()
	M_feat_mask = M_feat_mask_tens.numpy()

	M_feat_mask_tile = np.tile(np.expand_dims(M_feat_mask, 1), (1, map_c, 1, 1))

	T_feat_mask = np.multiply(T_feat, M_feat_mask_tile)

	r_m = T_feat_mask - M_feat_warp

	r_m_rsh = r_m.reshape((k, map_c * map_h * map_w, 1))

	return r_m_rsh


def compute_ri(I, T, V, P_fk, V_sz):
	'''
	:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		r_ba: Numpy array, residuals of images with templates, num_frame - 1 x sigma x (C x H x W) x 1
	'''

	sigma = V_sz.sum()

	num_frame, img_c, img_h, img_w = I.shape

	T_f = np.repeat(I[T, :, :, :], V_sz, axis=0)

	I_f = np.zeros((sigma, img_c, img_h, img_w))

	I_f_ind = 0
	for i in range(len(V)):
		for f in V[i]:
			I_f[I_f_ind, :, :, :] = I[f, :, :, :]
			I_f_ind = I_f_ind + 1

	P_kf_tens = torch.from_numpy(P_fk).float()
	I_f_tens = torch.from_numpy(I_f).float()

	I_f_warp_tens, I_f_mask_tens, _ = dlk.warp_hmg(I_f_tens, P_kf_tens)

	I_f_warp = I_f_warp_tens.numpy()
	I_f_mask = I_f_mask_tens.numpy()

	I_f_mask = np.tile(np.expand_dims(I_f_mask, 1), (1, img_c, 1, 1))

	T_f_mask = np.multiply(T_f, I_f_mask)

	r_ba = T_f_mask - I_f_warp

	r_ba = r_ba.reshape((sigma, img_c * img_h * img_w, 1))

	r_ba = np.tile(r_ba, (num_frame - 1, 1, 1, 1))

	return r_ba


def compute_Jm(Mf_gradx, Mf_grady, P_init, P, T, P_mk_rel, P_mk0):
	'''
	:
		Mf_gradx: Map image gradients in x, numpy array, k x Cf x Hf x Wf
		Mf_grady: Map image gradients in y, numpy array, k x Cf x Hf x Wf
		P_init: Absolute warp parameters relating map to first frame, 1 x 8 x 1
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates, length k
		P_mk_rel: Numpy array, warp parameters used on each of the map templates, k x 8 x 1
		P_mk0: Numpy array, warp parameters from the map to each map template, k x 8 x 1

	Returns:
		Jm: Numpy array, dM/dW * dW/dPmk * dPmk/dPF, num_frame - 1 x k x (Cf x Hf x Wf) x 8
	'''

	k_num = T.shape[0]
	num_frames = P.shape[0] + 1

	gradMf_warpjac = compute_gradI_warpjac(Mf_gradx, Mf_grady, P_mk_rel)
	gradMf_warpjac = np.tile(gradMf_warpjac, (num_frames - 1, 1, 1, 1))

	gradPmk_PF = np.zeros((num_frames - 1, k_num, 8, 8))

	for F in range(num_frames - 1):
		k_ind = 0
		for k in T:
			if (F < k):
				H_init = pama_utility.p_to_H(P_init)
				H_1k = pama_utility.p_to_H(P[0: k, :, :])
				H_mk = np.concatenate((
					H_init,
					H_1k), axis=0)

				H_F_ind = F + 1

				def Pmk(P_F):  # needs to calculate the derivative of P_F w.r.t. the actual image warp used on the given map template, since we are extracting the map template, and not resampling it every time

					H_F = pama_utility.p_to_H(P_F)

					H_mk_samp_temp = np.concatenate((
						H_mk[0: H_F_ind, :, :],
						H_F,
						H_mk[H_F_ind + 1:, :, :]
					), axis=0)

					# invert to go from sampling parameters to coordinate parameters
					H_mk_temp = np.linalg.inv(H_mk_samp_temp)

					# intialize the map->map-template warp as coord hmg
					H_mk_coord = np.eye(3)

					for i in range(H_mk_temp.shape[0]):
						H_mk_coord = np.dot(H_mk_temp[i, :, :], H_mk_coord)

					# compute the map-template -> map-template warp
					H_mk0_samp = pama_utility.p_to_H(P_mk0[k_ind, :, :])
					H_mk0_coord = np.linalg.inv(H_mk0_samp)

					H_warp_coord = H_mk_coord @ np.linalg.inv(H_mk0_coord)

					# invert back to sampling params
					H_warp_samp = np.linalg.inv(H_warp_coord)

					P_mk = pama_utility.H_to_p(H_warp_samp)
					P_mk = P_mk.squeeze(0)

					return P_mk

				grad_P_mk = jacobian(Pmk)

				P_F = pama_utility.H_to_p(H_mk[H_F_ind, :, :])
				P_F = P_F.squeeze(0)

				gradPmk_PF[F, k_ind, :, :] = \
					grad_P_mk(P_F).squeeze(axis=1).squeeze(axis=2)

			else:
				gradPmk_PF[F, k_ind, :, :] = np.zeros((8, 8))

			k_ind = k_ind + 1

	J_m = np.matmul(gradMf_warpjac, gradPmk_PF)

	return J_m


def compute_Ji(I_gradx, I_grady, P, T, V, P_fk, V_sz):
	'''
	:
		I_gradx: x-gradient of image sequence, 4D numpy array, num_frame x C x H x W
		I_grady: y-gradient of image sequence, 4D numpy array, num_frame X C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame - 1 x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		Ji: Numpy array, dI/dW * dW/dPfk * dPfk/dPF, num_frame - 1 x sigma x (C x H x W) x 8
	'''

	sigma = V_sz.sum()
	num_frame, img_c, img_h, img_w = I_gradx.shape

	I_gradx_f = np.zeros((sigma, img_c, img_h, img_w))
	I_grady_f = np.zeros((sigma, img_c, img_h, img_w))

	I_f_ind = 0
	for i in range(len(V)):
		for f in V[i]:
			I_gradx_f[I_f_ind, :, :, :] = I_gradx[f, :, :, :]
			I_grady_f[I_f_ind, :, :, :] = I_grady[f, :, :, :]
			I_f_ind = I_f_ind + 1

	gradI_warpjac = compute_gradI_warpjac(I_gradx_f, I_grady_f, P_fk)
	gradI_warpjac = np.tile(gradI_warpjac, (num_frame - 1, 1, 1, 1))

	gradPfk_PF = np.zeros((num_frame - 1, sigma, 8, 8))

	for F in range(num_frame - 1):
		frame_id = 0  # iterates through sigma axis
		k_ind = 0
		for k in T:
			for f in V[k_ind]:
				# f and k index images I, F indexes P
				if (f < k) and (F < k) and (f <= F):
					H_fk = pama_utility.p_to_H(P[f: k, :, :])

					H_F_ind = F - f

					def Pfk(P_F):
						H_F = pama_utility.p_to_H(P_F)

						H_fk_samp_temp = np.concatenate((
							H_fk[0:H_F_ind, :, :],
							H_F,
							H_fk[H_F_ind + 1:, :, :]
						), axis=0)

						# invert to go from sampling parameters to coordinate parameters
						H_fk_temp = np.linalg.inv(H_fk_samp_temp)

						H_fk_mat = np.eye(3)

						for i in range(H_fk_temp.shape[0]):
							H_fk_mat = np.dot(H_fk_temp[i, :, :], H_fk_mat)

						# after combining, invert back to sampling parameters
						H_fk_mat_samp = np.linalg.inv(H_fk_mat)
						P_fk = pama_utility.H_to_p(H_fk_mat_samp)
						P_fk = P_fk.squeeze(0)
						return P_fk

					# using auto-grad library for computing jacobian (8x8)
					grad_P_fk = jacobian(Pfk)

					P_F = pama_utility.H_to_p(H_fk[H_F_ind, :, :])
					P_F = P_F.squeeze(0)

					# evaluate 8x8 jacobian and store it
					gradPfk_PF[F, frame_id, :, :] = \
						grad_P_fk(P_F).squeeze(axis=1).squeeze(axis=2)

				elif (f > k) and (F >= k) and (f > F):
					H_kf = pama_utility.p_to_H(P[k: f, :, :])

					H_F_ind = F - k

					def Pfk(P_F):
						H_F = pama_utility.p_to_H(P_F)

						H_kf_temp_samp = np.concatenate((
							H_kf[0:H_F_ind, :, :],
							H_F,
							H_kf[H_F_ind + 1:, :, :]
						), axis=0)

						# invert to go from sampling parameters to coordinate parameters
						H_kf_temp = np.linalg.inv(H_kf_temp_samp)
						# but then, invert back because we are calculating are going backward from frame to template
						# therefore it is no-op, however leaving it here for explanation
						H_kf_temp_inv = np.linalg.inv(H_kf_temp)

						H_fk_mat = np.eye(3)

						for i in range(H_kf_temp_inv.shape[0]):
							H_fk_mat = np.dot(H_fk_mat, H_kf_temp_inv[i, :, :])

						# finally, invert from coordinate params back to sampling params
						H_fk_mat_samp = np.linalg.inv(H_fk_mat)
						P_fk = pama_utility.H_to_p(H_fk_mat_samp)
						P_fk = P_fk.squeeze(0)
						return P_fk

					grad_P_fk = jacobian(Pfk)

					P_F = pama_utility.H_to_p(H_kf[H_F_ind, :, :])
					P_F = P_F.squeeze(0)

					gradPfk_PF[F, frame_id, :, :] = \
						grad_P_fk(P_F).squeeze(axis=1).squeeze(axis=2)

				else:
					gradPfk_PF[F, frame_id, :, :] = np.zeros((8, 8))

				frame_id = frame_id + 1

			k_ind = k_ind + 1

	J = np.matmul(gradI_warpjac, gradPfk_PF)

	return J


def compute_Pmk(P_init, P, T):
	'''
	Args:
		P_init: Absolute warp parameters relating map to first frame, 1 x 8 x 1
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates, length k

	Returns:
		P_mk: Warp parameters from map to templates, k x 8 x 1
		p_mk的参数代表的是当前地图的绝对坐标，所以这里的P是传进来的参数，直接计算得到的，没有优化过程的参与
		不过这个函数里面进行的内容没怎么看懂，最后得到的应该是一个绝对的Pmk序列
		这个当T的带线啊哦等于0的时候返回值等于P_init，因此在sz2之中均为单位矩阵
		这里参数T给的定义是：T: Numpy vector containing indices from I which are templates, length k
		说明这里指的是T中包含的是关键帧的index
	'''

	P_mk = np.zeros((T.shape[0], 8, 1))

	k_ind = 0
	H_init = pama_utility.p_to_H(P_init)
	H_k = pama_utility.p_to_H(P[0, :, :])

	H_mk_samp = np.concatenate((
		H_init,
		H_k
	), axis=0)

	# invert to go from sampling params to coord params
	H_mk = np.linalg.inv(H_mk_samp)  # 矩阵的求逆操作

	# combine coord params
	H_mk_mat = np.eye(3)

	for i in range(H_mk.shape[0]):
		H_mk_mat = np.dot(H_mk[i, :, :], H_mk_mat)

	# after combining, invert back to sampling parameters
	H_mk_mat_samp = np.linalg.inv(H_mk_mat)
	P_mk[k_ind, :, :] = pama_utility.H_to_p(H_mk_mat_samp)
	return P_mk


def compute_Pfk(P, T, V, V_sz):
	'''
	:
		P: Warp parameter sequence, 3D numpy array, num_frame - 1 x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		Pfk: Numpy array, warp parameters from images to templates, sigma x 8 x 1
	'''

	sigma = V_sz.sum()

	P_fk_all = np.zeros((sigma, 8, 1))

	frame_id = 0  # iterates through sigma axis
	k_ind = 0
	for k in T:
		for f in V[k_ind]:
			# f and k index images I, F indexes P
			if (f < k):
				H_fk_samp = pama_utility.p_to_H(P[f: k, :, :])

				# invert to go from sampling parameters to coordinate parameters
				H_fk = np.linalg.inv(H_fk_samp)

				# combine coordinate warpings
				H_fk_mat = np.eye(3)

				for i in range(H_fk.shape[0]):
					H_fk_mat = np.dot(H_fk[i, :, :], H_fk_mat)

				# after combining, invert back to sampling parameters
				H_fk_mat_samp = np.linalg.inv(H_fk_mat)
				P_fk = pama_utility.H_to_p(H_fk_mat_samp)

			else:
				H_kf_samp = pama_utility.p_to_H(P[k: f, :, :])

				# invert to go from sampling parameters to coordinate parameters
				H_kf = np.linalg.inv(H_kf_samp)
				# but then, invert back because we are calculating are going backward from frame to template
				# therefore it is no-op, however leaving it here for explanation
				H_kf_inv = np.linalg.inv(H_kf)

				H_fk_mat = np.eye(3)

				for i in range(H_kf_inv.shape[0]):
					H_fk_mat = np.dot(H_fk_mat, H_kf_inv[i, :, :])

				# finally, invert from coordinate params back to sampling params
				H_fk_mat_samp = np.linalg.inv(H_fk_mat)
				P_fk = pama_utility.H_to_p(H_fk_mat_samp)

			P_fk_all[frame_id, :, :] = P_fk
			frame_id = frame_id + 1

		k_ind = k_ind + 1

	return P_fk_all


def compute_gradI_warpjac(I_gradx, I_grady, P_fk):
	'''
	:
		I_gradx: x-gradient of image sequence with duplicates representing vis. neighborhoods, 4D numpy array, sigma x C x H x W
		I_grady: y-gradient of image sequence with duplicates representing vis. neighborhoods, 4D numpy array, sigma X C x H x W
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1

	Returns:
		gradI_warpjac: Numpy array, dI/dW * dW/dPfk, sigma x (C x H x W) x 8
	'''

	batch_size, c, h, w = I_gradx.shape

	P_fk_tens = torch.from_numpy(P_fk).float()
	# 修改，添加向GPU的转化
	# P_fk_tens = P_fk_tens.cuda()
	I_gradx_tens = torch.from_numpy(I_gradx).float()
	I_grady_tens = torch.from_numpy(I_grady).float()

	img_gradx_w, _, _ = dlk.warp_hmg(I_gradx_tens, P_fk_tens)
	img_grady_w, _, _ = dlk.warp_hmg(I_grady_tens, P_fk_tens)

	img_gradx_w = img_gradx_w.view(batch_size, c * h * w, 1)
	img_grady_w = img_grady_w.view(batch_size, c * h * w, 1)

	x = torch.arange(w)
	y = torch.arange(h)
	X, Y = dlk.meshgrid(x, y)
	H_pq = dlk.param_to_H(P_fk_tens)
	xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel())), 0)
	xy = xy.repeat(batch_size, 1, 1)
	xy_warp = H_pq.bmm(xy)

	# extract warped X and Y, normalizing the homog coordinates
	X_warp = xy_warp[:, 0, :] / xy_warp[:, 2, :]
	Y_warp = xy_warp[:, 1, :] / xy_warp[:, 2, :]

	X_warp = X_warp.unsqueeze(dim=2)
	Y_warp = Y_warp.unsqueeze(dim=2)

	X_warp = X_warp.repeat(1, c, 1)
	Y_warp = Y_warp.repeat(1, c, 1)

	gradI_warpjac = torch.cat((
		X_warp.mul(img_gradx_w),
		Y_warp.mul(img_gradx_w),
		img_gradx_w,
		X_warp.mul(img_grady_w),
		Y_warp.mul(img_grady_w),
		img_grady_w,
		-X_warp.mul(X_warp).mul(img_gradx_w) - X_warp.mul(Y_warp).mul(img_grady_w),
		-X_warp.mul(Y_warp).mul(img_gradx_w) - Y_warp.mul(Y_warp).mul(img_grady_w)), 2)

	return gradI_warpjac.numpy()


def extract_map_templates(P_mk, M, T, img_c, img_h, img_w):
	# 从基准图中抠出大图
	k_num = T.shape[0]

	_, map_h, map_w = M.shape

	M_tens = torch.from_numpy(M).unsqueeze(0).float()

	M_tmpl = np.zeros((k_num, img_c, img_h, img_w))

	if (map_w / map_h) > ( img_w / img_h):
		aspect = img_w / img_h

		adj_img_w = round(aspect * map_h)  # 计算符合长宽比的基准图宽度

		# bounding box for crop, centered in middle of map coordinate system
		# 使得这两张图比例一致
		left = round(map_w / 2 - adj_img_w / 2)
		upper = 0
		right = round(map_w / 2 + adj_img_w / 2)
		lower = map_h
	else:
		aspect = img_h / img_w

		adj_img_h = round(aspect * map_w)  # 计算符合长宽比的基准图宽度

		# bounding box for crop, centered in middle of map coordinate system
		# 使得这两张图比例一致
		left = 0
		upper = round(map_h / 2 - adj_img_h / 2)
		right = map_w
		lower = round(map_h / 2 + adj_img_h / 2)

	k_ind = 0

	# plt.figure()
	print('extracting map templates...')

	for k in T:
		P_mk_tens = torch.from_numpy(P_mk[k_ind: k_ind + 1, :, :]).float()
		# 修改
		P_mk_tens = P_mk_tens.to(device)
		# P_mk_tens = P_mk_tens.cuda()

		# no need to save mask, since the images extracted from the map (should) be valid at all pixels
		# 先做Warp变换，之后进行crop裁剪
		# 这里的warping操作问题比较多，是patch定位的关键所在
		# 计算出M_warp的中心坐标在原图中的位置，我们即可得到patch在M中的位置
		M_warp_tens, _, xy_patch_curr_cor = dlk.warp_hmg(M_tens, P_mk_tens)

		M_warp_pil = transforms.ToPILImage()(M_warp_tens.squeeze(0))

		# crop image segment matching template
		M_warp_pil_crop = M_warp_pil.crop((left, upper, right, lower))

		# 修改
		# 对于图片的显示，这里面出现了什么情况
		# 具体运行时可以注释掉
		# M_pil = transforms.ToPILImage()(torch.from_numpy(M))
		# M_pil_marked = srh.Add_M_Marker(M_pil, xy_patch_org_cor)
		# plt.imshow(M_pil_marked)
		# plt.show()
		# plt.subplot(2, 1, 2)
		# plt.imshow(M_warp_pil_crop)
		# plt.subplot(2, 1, 1)
		# plt.imshow(M_warp_pil)
		# plt.show()

		M_warp_pil_sm = M_warp_pil_crop.resize((img_w, img_h), Image.BILINEAR)

		M_warp_np_sm = transforms.ToTensor()(M_warp_pil_sm)

		M_tmpl[k_ind, :, :, :] = M_warp_np_sm

		print('... {:d}/{:d} ...'.format(k_ind + 1, k_num))

		k_ind = k_ind + 1

	# 找到了M图中与template所对应的patch
	return M_tmpl, xy_patch_curr_cor


def sliding_window_opt():
	xy_cor_list = []
	xy_cor_list_opt = []
	xy_cor_list_gt = []
	distance_error_list = []
	diff_score_list = []

	deep_net = dlk.custom_net(model_path)

	# opt_img_height参数决定了实时图与基准图之间的分辨率大小
	I_org, GPS_list = img_utility.load_I(image_dir, image_dir_ext, opt_img_height)
	_, img_c, img_h, img_w = I_org.shape

	M = img_utility.load_M(map_loc)
	_, map_h, map_w = M.shape

	# 这里的P参数是需要我们传进来的
	P_init_org, P_org = pama_utility.load_P(motion_param_loc)
	P_opt = np.zeros(P_org.shape)
	curr_P_init = P_init_org

	# 就是离线跑的，人为这里的循环次数就是P_org的长度
	for i in range(P_org.shape[0]):
		start_timestamp = time.clock()
		I = I_org[i: i + 2, :, :, :]

		# t,V是之后优化中会用到的参数，这里进行定义
		# templates indices
		T = np.array([1])
		# visibility neighborhood
		V = np.array(
			[
				np.arange(50),
				np.array([0])
			]
		)

		V_ = np.array([np.array([0])])
		V = np.delete(V, 0)

		# curr_P shape [1,8,1]
		curr_P = np.expand_dims(P_org[i, :, :], axis=0)
		# 这里的改变是让整个矩阵的转移大小可以符合我们的要求
		curr_P = pama_utility.scale_P(curr_P, s=scale_img_to_map)
		P_mk = compute_Pmk(curr_P_init, curr_P, T)

		# 采用瀑布流的形式（从粗到细，进行结果的细化
		for idx in range(2):
			# 从基准地图中获取信息的步骤
			# 同时增加一个参数，接收其坐标值
			M_tmpl, xy_cor = extract_map_templates(P_mk, M, T, img_c, img_h, img_w)
			xy_cor_list.append(xy_cor)

			###
			# 提取得到M图中的patch之后，再结合实时图（tamplate frame）去提取两者之间的特征，进而进一步进行单应矩阵的细调
			T_np = np.expand_dims(I[1, :, :, :], axis=0)
			T_tens = Variable(torch.from_numpy(T_np).float())
			T_tens_nmlz = dlk.normalize_img_batch(T_tens)
			T_feat_tens = deep_net(T_tens_nmlz)
			T_feat = T_feat_tens.data.numpy()

			M_tmpl_tens = Variable(torch.from_numpy(M_tmpl).float())
			M_tmpl_tens_nmlz = dlk.normalize_img_batch(M_tmpl_tens)
			M_feat_tens = deep_net(M_tmpl_tens_nmlz)
			M_feat = M_feat_tens.data.numpy()

			###
			# 使用dlk的纠正手段
			dlk_net = dlk.DeepLK(dlk.custom_net(model_path))
			p_lk, _, itr_dlk = dlk_net(M_tmpl_tens_nmlz, T_tens_nmlz, tol=1e-4, max_itr=max_itr_dlk, conv_flag=1, ret_itr=True)
			# 计算patch与实时图之间的差距
			diff_score = torch.sqrt(torch.sum(torch.pow(M_tmpl_tens_nmlz-T_tens_nmlz, 2))).item()
			diff_score = diff_score / (M_tmpl_tens.shape[1]*M_tmpl_tens.shape[2]*M_tmpl_tens.shape[3])
			print("基准子图与实时图之间的差距为:", diff_score)
			diff_score_list.append(diff_score)
			p_lk = p_lk.cpu()
			p_lk2x = pama_utility.scale_P(p_lk.data.numpy(), scaling_for_disp)
			# 只使用VO的结果
			# p_lk2x = np.zeros([1, 8, 1])

			s_sm = (scaling_for_disp * opt_img_height) / map_h
			curr_P_init_scale = pama_utility.scale_P(curr_P_init, s_sm)
			curr_P_scale = pama_utility.scale_P(curr_P, s_sm)

			H_rel_samp = pama_utility.p_to_H(p_lk2x)
			H_org_samp = pama_utility.p_to_H(curr_P_scale)
			H_rel_coord = np.linalg.inv(H_rel_samp)
			H_org_coord = np.linalg.inv(H_org_samp)
			H_opt_coord = H_rel_coord @ H_org_coord
			H_opt_samp = np.linalg.inv(H_opt_coord)

			P_opt_i_scale = pama_utility.H_to_p(H_opt_samp)
			s_lg = map_h / (scaling_for_disp * opt_img_height)
			P_opt_i = pama_utility.scale_P(P_opt_i_scale, s_lg)
			P_opt[i, :, :] = P_opt_i

			P_mk_opt = compute_Pmk(curr_P_init_scale, P_opt_i_scale, T)
			P_mk0 = compute_Pmk(curr_P_init_scale, curr_P_scale, T)
			P_mk_opt_map = pama_utility.scale_P(P_mk_opt, s_lg)
			P_mk0_map = pama_utility.scale_P(P_mk0, s_lg)
			H_mk_samp = pama_utility.p_to_H(P_mk_opt)
			H_mk0_samp = pama_utility.p_to_H(P_mk0)

			H_mk = np.linalg.inv(H_mk_samp)
			H_mk0 = np.linalg.inv(H_mk0_samp)
			H_mk_rel = np.matmul(H_mk, np.linalg.inv(H_mk0))
			H_mk_rel_samp = np.linalg.inv(H_mk_rel)
			P_mk_rel_samp = pama_utility.H_to_p(H_mk_rel_samp)
			H_init_samp = pama_utility.p_to_H(curr_P_init)
			H_opt_i_samp = pama_utility.p_to_H(P_opt_i)

			H_init_coord = np.linalg.inv(H_init_samp)
			H_opt_i_coord = np.linalg.inv(H_opt_i_samp)
			H_init_coord_new = H_opt_i_coord @ H_init_coord
			H_init_samp_new = np.linalg.inv(H_init_coord_new)
			curr_P_init = pama_utility.H_to_p(H_init_samp_new)

			P_mk = curr_P_init
			curr_P = torch.zeros([1,8,1])

		# need to rescale P_init_org and P_org to opt_img_height, from map_h
		# s_sm = opt_img_height / map_h;
		# s_sm = (scaling_for_disp * opt_img_height) / map_h  #########################################
		# curr_P_init_scale = pama_utility.scale_P(curr_P_init, s_sm)
		# curr_P_scale = pama_utility.scale_P(curr_P, s_sm)
		#
		# # 这部分的内容是在增加优化部分的内容
		# # 直接注释掉这一行就是不使用优化跑的结果
		# # P_opt_i_scale = optimize_wmap(I, curr_P_scale, T, V, curr_P_init_scale, M_feat, T_feat, tol, max_itr, lam1, lam2)
		# # curr_P_scale = optimize_wmap(I_2x, curr_P_scale, T, V, curr_P_init_scale, M_feat_2x, T_feat_2x, tol, max_itr, lam1, lam2)
		#
		# H_rel_samp = pama_utility.p_to_H(p_lk2x)
		# H_org_samp = pama_utility.p_to_H(curr_P_scale)
		# H_rel_coord = np.linalg.inv(H_rel_samp)
		# H_org_coord = np.linalg.inv(H_org_samp)
		# H_opt_coord = H_rel_coord @ H_org_coord
		# H_opt_samp = np.linalg.inv(H_opt_coord)
		#
		# P_opt_i_scale = pama_utility.H_to_p(H_opt_samp)
		# s_lg = map_h / (scaling_for_disp * opt_img_height)
		# P_opt_i = pama_utility.scale_P(P_opt_i_scale, s_lg)
		# P_opt[i, :, :] = P_opt_i
		#
		# ### plotting
		# P_mk_opt = compute_Pmk(curr_P_init_scale, P_opt_i_scale, T)
		# P_mk0 = compute_Pmk(curr_P_init_scale, curr_P_scale, T)
		# P_mk_opt_map = pama_utility.scale_P(P_mk_opt, s_lg)
		# P_mk0_map = pama_utility.scale_P(P_mk0, s_lg)
		# H_mk_samp = pama_utility.p_to_H(P_mk_opt)
		# H_mk0_samp = pama_utility.p_to_H(P_mk0)
		#
		# # invert sampling params to get coord params:
		# H_mk = np.linalg.inv(H_mk_samp)
		# H_mk0 = np.linalg.inv(H_mk0_samp)
		#
		# # compute relative hmg:
		# H_mk_rel = np.matmul(H_mk, np.linalg.inv(H_mk0))
		#
		# # invert back to sampling hmg:
		# H_mk_rel_samp = np.linalg.inv(H_mk_rel)
		#
		# # convert sampling hmg back to sampling params:
		# P_mk_rel_samp = pama_utility.H_to_p(H_mk_rel_samp)
		#
		#
		# # compose new P_opt_i with curr_P_init
		# H_init_samp = pama_utility.p_to_H(curr_P_init)
		# H_opt_i_samp = pama_utility.p_to_H(P_opt_i)
		#
		# H_init_coord = np.linalg.inv(H_init_samp)
		# H_opt_i_coord = np.linalg.inv(H_opt_i_samp)
		#
		# H_init_coord_new = H_opt_i_coord @ H_init_coord
		#
		# H_init_samp_new = np.linalg.inv(H_init_coord_new)
		#
		# curr_P_init = pama_utility.H_to_p(H_init_samp_new)



		end_timestamp = time.clock()
		print("一张图片定位所需时间为：", end_timestamp-start_timestamp)
		print('finished iteration: {:d}'.format(i + 1))

		# 定位过程结束，以下部分是画图部分的内容
		side_margin = 0.15
		top_margin = 0.15
		targ_box = np.array([
			[M_tmpl_tens.shape[3] * side_margin, M_tmpl_tens.shape[2] * top_margin],
			[M_tmpl_tens.shape[3] * (1 - side_margin), M_tmpl_tens.shape[2] * top_margin],
			[M_tmpl_tens.shape[3] * (1 - side_margin), M_tmpl_tens.shape[2] * (1 - side_margin)],
			[M_tmpl_tens.shape[3] * side_margin, M_tmpl_tens.shape[2] * (1 - side_margin)],
			[M_tmpl_tens.shape[3] * side_margin, M_tmpl_tens.shape[2] * side_margin],
			[M_tmpl_tens.shape[3] * (1 - side_margin), M_tmpl_tens.shape[2] * (1 - side_margin)],
		])

		plt.subplot(3, 1, 1)
		plt.title('M{:d}'.format(i + 1))
		plt.imshow(img_utility.plt_axis_match_np(M_tmpl_tens[0, :, :, :]))
		plt.plot(targ_box[:, 0], targ_box[:, 1], 'r-')
		plt.plot(round(M_tmpl_tens.shape[3]/2),round(M_tmpl_tens.shape[2]/2),'ro')
		plt.axis('off')

		plt.subplot(3, 1, 2)
		plt.title('M{:d} Warp'.format(i + 1))
		M_tmpl_curr_tens = M_tmpl_tens.float()
		P_mk_rel_samp_curr = torch.from_numpy(pama_utility.scale_P(P_mk_rel_samp, scaling_for_disp)).float()
		M_tmpl_w, _, xy_cor_curr_opt = dlk.warp_hmg(M_tmpl_curr_tens, P_mk_rel_samp_curr)
		# 进一步优化的内容
		# xy_cor_opt变量中存的是新的细化后的图像在原图像（上一次warp+crop之后的图像）中坐标位置
		print("未经过优化的绝对坐标为：", xy_cor)
		# print("经过优化的相对的坐标为：", xy_cor_curr_opt)
		xy_patch_org_cor_opt = dlk.warp_hmg_Noncentric(M, P_mk, xy_cor_curr_opt, img_w=T_tens.shape[3], img_h=T_tens.shape[2])
		xy_cor_list_opt.append(xy_patch_org_cor_opt)
		print("经过优化的绝对的坐标为：", xy_patch_org_cor_opt)
		xy_cor_curr_gt = srh.lag_log_to_pix_pos(M, GPS_list[i+1])
		xy_cor_list_gt.append(xy_cor_curr_gt)
		print("实际坐标为：", [xy_cor_curr_gt[0], xy_cor_curr_gt[1]])
		distance_error_curr = map_resolution * sqrt((xy_cor_curr_gt[0]-xy_patch_org_cor_opt[0])**2 + (xy_cor_curr_gt[1]-xy_patch_org_cor_opt[1])**2)
		print("实际坐标与定位结果之间的距离：", distance_error_curr, "m")
		distance_error_list.append(distance_error_curr)

		plt.imshow(img_utility.plt_axis_match_tens(M_tmpl_w[0, :, :, :]))
		# plt.plot(M_tmpl_w.shape[3]/2, M_tmpl_w.shape[2]/2, 'ro')
		plt.plot(targ_box[:, 0], targ_box[:, 1], 'r-')
		plt.plot(round(M_tmpl_tens.shape[3] / 2), round(M_tmpl_tens.shape[2] / 2), 'ro')
		plt.axis('off')

		plt.subplot(3, 1, 3)
		plt.imshow(img_utility.plt_axis_match_np(I_org[i + 1, :, :, :]))
		# plt.plot(I_org_2x.shape[3]/2, I_org_2x.shape[2]/2, 'ro')
		plt.plot(targ_box[:, 0], targ_box[:, 1], 'r-')
		plt.plot(round(M_tmpl_tens.shape[3] / 2), round(M_tmpl_tens.shape[2] / 2), 'ro')
		plt.title('I{:d}'.format(i + 1))
		plt.axis('off')

		# 可以先不显示这里每一步的warp过程
		plt.show()

		# 	# 画一下优化对比的结果
		# 	# 这一步是比较耗时的一步，基本上去掉这一步之后可以达到实时运行的效果了
		# if i == 60:
		# 	plt.subplot(2, 1, 1)
		# 	M_pil_marked = srh.Add_M_Marker(transforms.ToPILImage()(torch.from_numpy(M)), xy_patch_org_cor_opt,
		# 									color="blue")
		# 	M_pil_marked = srh.Add_M_Marker(M_pil_marked, xy_cor, color="red")
		# 	M_pil_marked = srh.Add_M_Marker(M_pil_marked, xy_cor_curr_gt, color="green")
		# 	plt.title('当前定位结果，优化前：红；优化后：蓝；实际点：绿')
		# 	plt.imshow(M_pil_marked)
		#
		# 	plt.subplot(2, 1, 2)
		# 	M_pil_marked = srh.Add_M_Markers_list(M, xy_cor_list, color="red")
		# 	M_pil_marked = srh.Add_M_Markers_list(M_pil_marked, xy_cor_list_opt, color="blue")
		# 	M_pil_marked = srh.Add_M_Markers_list(M_pil_marked, xy_cor_list_gt, color="green")
		# 	plt.title('飞行轨迹历史，优化前：红；优化后：蓝；实际点：绿')
		# 	plt.imshow(M_pil_marked)
		# 	plt.show()
		#
		# 	plt.plot(diff_score_list)
		# 	plt.show()


	plt.subplot(2, 1, 1)
	M_pil_marked = srh.Add_M_Marker(transforms.ToPILImage()(torch.from_numpy(M)), xy_patch_org_cor_opt,
									color="blue")
	M_pil_marked = srh.Add_M_Marker(M_pil_marked, xy_cor, color="red")
	M_pil_marked = srh.Add_M_Marker(M_pil_marked, xy_cor_curr_gt, color="green")
	plt.title('当前定位结果，优化前：红；优化后：蓝；实际点：绿')
	plt.imshow(M_pil_marked)

	plt.subplot(2, 1, 2)
	M_pil_marked = srh.Add_M_Markers_list(M, xy_cor_list, color="red")
	M_pil_marked = srh.Add_M_Markers_list(M_pil_marked, xy_cor_list_opt, color="blue")
	M_pil_marked = srh.Add_M_Markers_list(M_pil_marked, xy_cor_list_gt, color="green")
	plt.title('飞行轨迹历史，优化前：红；优化后：蓝；实际点：绿')
	plt.imshow(M_pil_marked)
	plt.show()

	plt.plot(diff_score_list)
	plt.show()
	plt.plot(distance_error_list)
	plt.show()
	M_pil_marked.save('marked.png')
	draw = ImageDraw.Draw(M_pil_marked)
	for i in range(len(xy_cor_list_opt)-1):
		draw.line((xy_cor_list_opt[i][0], xy_cor_list_opt[i][1], xy_cor_list_opt[i+1][0], xy_cor_list_opt[i+1][1]), fill='blue',width=5)
		draw.line((xy_cor_list[i][0], xy_cor_list[i][1], xy_cor_list[i + 1][0], xy_cor_list[i + 1][1]), fill='red',width=5)
		draw.line((xy_cor_list_gt[i][0], xy_cor_list_gt[i][1], xy_cor_list_gt[i + 1][0], xy_cor_list_gt[i + 1][1]), fill='green',width=5)
	M_pil_marked.save('marked_line.png')
	# s_rel_pose = float(img_h_rel_pose) / map_h
	s_rel_pose = 1
	P_opt_scale = pama_utility.scale_P(P_opt, s_rel_pose)
	H_opt_rel_samp = pama_utility.p_to_H(P_opt_scale)
	H_opt_rel_coord = np.linalg.inv(H_opt_rel_samp)

	P_opt_map_scale_coord = pama_utility.H_to_p(H_opt_rel_coord)

	P_opt_map_scale_coord_sqz = np.squeeze(P_opt_map_scale_coord)

	# switch axes in order to have proper format for next function, decompose_rel_hmg.py
	H_opt_rel_coord = H_opt_rel_coord.swapaxes(0, 2)
	H_opt_rel_coord = H_opt_rel_coord.swapaxes(0, 1)
	print("平均精度为：", np.sum(np.array(distance_error_list))/17)

	sio.savemat(opt_param_save_loc, dict([('H_rel', H_opt_rel_coord),
										  ('cor_opt', xy_cor_list_opt),
										  ('diff_score', diff_score_list)]))


def main():
	# 一些为了显示更加方便添加的语句
	warnings.filterwarnings("ignore")
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	np.set_printoptions(precision=4)
	if (test == 'wmap'):
		optimize_wmap()
	elif (test == 'sliding_window'):
		sliding_window_opt()


if __name__ == "__main__":
	main()
