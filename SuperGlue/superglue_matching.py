import argparse
import cv2
import matplotlib.cm as cm
import torch
import sys

sys.path.append('./SuperGlue/')

from models.matching import Matching
from models.utils import make_matching_plot_fast

torch.set_grad_enabled(False)

def superglue_load(opt):
	device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'

	config = {
		'superpoint': {
		    'nms_radius': opt.nms_radius,
		    'keypoint_threshold': opt.keypoint_threshold,
		    'max_keypoints': opt.max_keypoints
		},
		'superglue': {
		    'weights': opt.superglue,
		    'sinkhorn_iterations': opt.sinkhorn_iterations,
		    'match_threshold': opt.match_threshold,
		}
	}
	matching = Matching(config).eval().to(device)
	return matching, device

def superglue_matching(matching, img1, img2, device, invalid=True):
	img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	img1_ = torch.from_numpy(img1_ / 255.0)[None, None].float().to(device)
	img2_ = torch.from_numpy(img2_ / 255.0)[None, None].float().to(device)

	pred = matching({"image0": img1_, "image1": img2_}, invalid)

	kpts0 = pred['keypoints0'][0].cpu().numpy()
	kpts1 = pred['keypoints1'][0].cpu().numpy()
	matches = pred['matches0'][0].cpu().numpy()
	confidence = pred['matching_scores0'][0].cpu().numpy()
	
	valid = matches > -1
	mkpts0 = kpts0[valid]
	mkpts1 = kpts1[matches[valid]]
	color = cm.jet(confidence[valid])
	text = [
		'SuperGlue',
		'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
		'Matches: {}'.format(len(mkpts0))
	]
	k_thresh = matching.superpoint.config['keypoint_threshold']
	m_thresh = matching.superglue.config['match_threshold']
	small_text = [
		'Keypoint Threshold: {:.4f}'.format(k_thresh),
		'Match Threshold: {:.2f}'.format(m_thresh),
	]
	out = make_matching_plot_fast(
		img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
		path=None, show_keypoints=True, small_text=small_text)

	return mkpts0, mkpts1, out   
