import numpy as np
import cv2
import os 
import sys
sys.path.append("./SuperGlue/")
from superglue_matching import superglue_load, superglue_matching
from matching_utils import ExtractFeatures, computeNN, sift_matching, crop_img, draw_matching, img_processing
from config import get_config, print_usage

def registration(opt):
	Extractor = ExtractFeatures(num_kpt=opt.num_kpt, contrastThreshold=opt.contrastThreshold)
	matching, device = superglue_load(opt)

	img1 = cv2.imread(opt.img1_path)
	img2 = cv2.imread(opt.img2_path)

	img1 = crop_img(img1, opt.size)
	img2 = crop_img(img2, opt.size)
	
	## Laplacian if needed, but we found it is unhelpful
	if opt.laplacian == True:
		img1 = img_processing(img1).astype(np.uint8())
		img2 = img_processing(img2).astype(np.uint8())
	
	if opt.method == 'SIFT':
		## SIFT
		kpts1, kpts2, vis = sift_matching(Extractor, img1, img2, opt.rt_thr)
	
	elif opt.method == 'SuperGlue':
		## SuperPoint + SuperGlue
		kpts1, kpts2, vis = superglue_matching(matching, img1, img2, device, invalid=False)
	
	else:
		raise RuntimeError('Unsupported matching method')

	print('Initial Matching: %03d' % (kpts1.shape[0]))
	cv2.imwrite('initial_matching.jpg', vis)	

	if kpts1.shape[0] < 3:
		print('Alignment failed')
		return 0
	
	## Affine transformation
	Mat, mask = cv2.estimateAffine2D(kpts1, kpts2, cv2.RANSAC, ransacReprojThreshold=opt.ransac_thr)

	mask = mask.squeeze(-1)
	
	## Select inliers for initial matches
	kpts1 = kpts1[mask == 1]
	kpts2 = kpts2[mask == 1]
	
	## Reprojection error estimation
	proj_kpts1 = np.matmul(Mat, np.concatenate([kpts1, np.ones((kpts1.shape[0], 1))], axis=-1).T).T
	proj_err = np.linalg.norm(kpts2 - proj_kpts1, ord=2, axis=-1)
	proj_err_mean = np.mean(proj_err)
	proj_err_std = np.std(proj_err)

	print('Inliers: %03d --- Reproj Err mean: %.2f --- std: %.2f' % (kpts1.shape[0], proj_err_mean, proj_err_std))

	vis = draw_matching(img1, img2, kpts1, kpts2, thickness=2)
	cv2.imwrite('inliers.jpg', vis)
	
	## Warp source image based on affine transformation
	warped_img = cv2.warpAffine(img1, Mat, (img2.shape[1], img2.shape[0]))

	## Fuse warped image and target image
	warped_img = 0.5 * warped_img + 0.5 * img2

	cv2.imwrite('warped_img.jpg', warped_img)

if __name__ == '__main__':
	opt, _ = get_config()
	print_usage()
	registration(opt)		
