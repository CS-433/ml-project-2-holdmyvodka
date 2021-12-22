import numpy as np
import cv2
from imutils import adjust_brightness_contrast
import sys

class ExtractFeatures(object):
	def __init__(self, num_kpt, contrastThreshold=1e-5):
		self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kpt, contrastThreshold=contrastThreshold)
		self.num_kpt = num_kpt
	
	def run(self, img):
		if len(img.shape) > 2:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		cv_kp_sift, desc_sift = self.sift.detectAndCompute(img, None)
	
		kp_sift = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp_sift])
		
		## remove keypoints located on the background for task1 and task2
		mask = (1 - (kp_sift[:, 0] > 800) * (kp_sift[:, 1] < 230))
		kp_sift = kp_sift[mask != 0]
		desc_sift = desc_sift[mask != 0]
		
		return kp_sift, desc_sift

def computeNN(desc_ii, desc_jj):
	d_ii = (desc_ii**2).sum(axis=1)
	d_jj = (desc_jj**2).sum(axis=1)
	distmat = (d_ii[:, None] + d_jj[None] - 2 * np.matmul(desc_ii, desc_jj.T))
	distmat = np.sqrt(distmat)
	
	distVals = np.sort(distmat, axis=1)[:, :2]
	nnIdx1 = np.argsort(distmat, axis=1)[:, :2]
	nnIdx1 = nnIdx1[:, 0]
	
	nnIdx2 = np.argsort(distmat, axis=0)[0, :]
	
	## mutual nearest test
	mutual_nearest = (nnIdx2[nnIdx1] == np.arange(0, nnIdx1.shape[0]))
	
	## ratio test
	ratio_test = (distVals[:, 0] / distVals[:, 1].clip(min=1e-10))
	
	idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1]
	
	return idx_sort, ratio_test, mutual_nearest

def crop_img(img, dim=224):
	bbx = img.sum(axis=-1)
	bbx = np.where(bbx > 0)
	bbx = [bbx[0].min(), bbx[1].min(), bbx[0].max(), bbx[1].max()]
	img = img[bbx[0]:bbx[2], bbx[1]:bbx[3]]

	padding = max(img.shape[0], img.shape[1])
	top_pad = int((padding - img.shape[0]) / 2)
	bottom_pad = int((padding - img.shape[0]) / 2)
	left_pad = int((padding - img.shape[1]) / 2)
	right_pad = int((padding - img.shape[1]) / 2)

	img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
	img = cv2.resize(img, (dim, dim))
	return img

def sift_matching(Extractor, img1, img2, rt_thr):
	## detect keypoints and describe features
	kpts1, desc1 = Extractor.run(img1)
	kpts2, desc2 = Extractor.run(img2)
	
	## brute force matching
	idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)
	
	kpts2 = kpts2[idx_sort[1], :]
	
	## pre-remove false matches
	kpts1 = kpts1[mutual_nearest * (ratio_test < rt_thr)]
	kpts2 = kpts2[mutual_nearest * (ratio_test < rt_thr)]

	vis = draw_matching(img1, img2, kpts1, kpts2, thickness=1)
	
	return kpts1, kpts2, vis

def adjust_luminance(img):
	img = adjust_brightness_contrast(img, brightness=0.0, contrast=0.0)	
	w_r = 0.05
	w_g = 0.9
	w_b = 0.05
	img = w_b * img[:, :, 0] + w_g * img[:, :, 1] + w_r * img[:, :, 2]	
	return img

def gamma_correlation(img, gamma=0.5, gain=1.0):
	img = 255 * gain * (img / 255.) ** (gamma)
	return img

def laplacian(img, kernel_size=3):
	img = cv2.Laplacian(img.astype(np.uint8()), cv2.CV_64F, ksize=kernel_size)
	img = cv2.convertScaleAbs(img)
	return img

def binarization(img):
	img = cv2.blur(img, (3, 3), 0)
	thr, _ = cv2.threshold(img.astype(np.uint8()), 0, 255, cv2.THRESH_OTSU)
	thr, img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
	return img

def Sobel_Edge_Detection(img):
	img = cv2.blur(img, (5, 5), 0)
	sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=7)
	return sobelxy

def img_processing(img):
	mask = (img.sum(axis=-1) > 10)
	img = adjust_luminance(img)
	img = Sobel_Edge_Detection(img)
	img = img * mask
	img = binarization(img)
	return img


def draw_matching(img1, img2, kpts1, kpts2, thickness=1):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	
	if len(img1.shape) == 3:
		vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
	else:
		vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
	
	vis[:h1, :w1] = img1
	vis[:h2, w1: w1 + w2] = img2	
	
	green = (0, 255, 0)
	red = (0, 0, 255)

	for i in range(kpts1.shape[0]):
		x1 = int(kpts1[i, 0])
		y1 = int(kpts1[i, 1])
		x2 = int(kpts2[i, 0] + w1)
		y2 = int(kpts2[i, 1])
		
		cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))
	return vis
