import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue

class Matching(torch.nn.Module):
	""" Image Matching Frontend (SuperPoint + SuperGlue) """
	def __init__(self, config={}):
		super().__init__()
		self.superpoint = SuperPoint(config.get('superpoint', {}))
		self.superglue = SuperGlue(config.get('superglue', {}))
	
	def remove_invalid_kpts(self, pred):
		mask = 1 - (pred["keypoints"][0][:, 0] > 800).float() * (pred["keypoints"][0][:, 1] < 230).float()
		mask_indices = torch.nonzero(mask).squeeze(-1)
		scores = torch.from_numpy(pred["scores"][0].numpy()).to(mask.device)[mask_indices]

		pred["keypoints"][0] = pred["keypoints"][0][mask_indices]
		pred["descriptors"][0] = pred["descriptors"][0][:, mask_indices]
		pred["scores"] = [scores]
		return pred

	def forward(self, data, invalid=True):
		""" Run SuperPoint (optionally) and SuperGlue
		SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
		Args:
		  data: dictionary with minimal keys: ['image0', 'image1']
		"""
		pred = {}

		# Extract SuperPoint (keypoints, scores, descriptors) if not provided
		if 'keypoints0' not in data:
			pred0 = self.superpoint({'image': data['image0']})
			pred0 = self.remove_invalid_kpts(pred0) if invalid is True else pred0
			pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
		if 'keypoints1' not in data:
			pred1 = self.superpoint({'image': data['image1']})
			pred1 = self.remove_invalid_kpts(pred1) if invalid is True else pred1
			pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

		# Batch all features
		# We should either have i) one image per batch, or
		# ii) the same number of local features for all images in the batch.
		data = {**data, **pred}

		for k in data:
			if isinstance(data[k], (list, tuple)):
				data[k] = torch.stack(data[k])

		# Perform the matching
		pred = {**pred, **self.superglue(data)}

		return pred
