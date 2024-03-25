"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07, N=50020,
				 base_temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.base_temperature = base_temperature

		self.u = torch.zeros(N).reshape(-1, 1)
		self.v = torch.zeros(N)
		self.f1_w = torch.zeros(N).reshape(-1, 1)
		self.f2_w = torch.zeros(N)
	
	# def forward(self, features, index, labels=None, mask=None, gamma=0.9):
	# 	"""Compute loss for model. If both `labels` and `mask` are None,
	# 	it degenerates to SimCLR unsupervised loss:
	# 	https://arxiv.org/pdf/2002.05709.pdf

	# 	Args:
	# 		features: hidden vector of shape [bsz, n_views, ...].
	# 		labels: ground truth of shape [bsz].
	# 		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
	# 			has the same class as sample i. Can be asymmetric.
	# 	Returns:
	# 		A loss scalar.
	# 	"""
	# 	device = (torch.device('cuda')
	# 			  if features.is_cuda
	# 			  else torch.device('cpu'))
	
	# 	if len(features.shape) < 3:
	# 		raise ValueError('`features` needs to be [bsz, n_views, ...],'
	# 						 'at least 3 dimensions are required')
	# 	if len(features.shape) > 3:
	# 		features = features.view(features.shape[0], features.shape[1], -1)

	# 	batch_size = features.shape[0]
	# 	if labels is not None and mask is not None:
	# 		raise ValueError('Cannot define both `labels` and `mask`')
	# 	elif labels is None and mask is None:
	# 		mask = torch.eye(batch_size, dtype=torch.float32).to(device)
	# 	elif labels is not None:
	# 		labels = labels.contiguous().view(-1, 1)
	# 		if labels.shape[0] != batch_size:
	# 			raise ValueError('Num of labels does not match num of features')
	# 		mask = torch.eq(labels, labels.T).float().to(device)
	# 	else:
	# 		mask = mask.float().to(device)

	# 	contrast_count = features.shape[1]
	# 	contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
	
	# 	anchor_feature = features[:, 0]
	# 	anchor_count = 1
		
	# 	# compute logits
	# 	anchor_dot_contrast = torch.div(
	# 		torch.matmul(anchor_feature, contrast_feature.T),
	# 		self.temperature)
		
	# 	logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		
	# 	logits = anchor_dot_contrast - logits_max.detach()

	# 	# tile mask
	# 	mask = mask.repeat(anchor_count, contrast_count)
		
	# 	# mask-out self-contrast cases
	# 	logits_mask = torch.scatter(
	# 		torch.ones_like(mask),
	# 		1,
	# 		torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
	# 		0
	# 	)
		
	# 	mask = mask * logits_mask

	# 	# compute log_prob
	# 	exp_logits = torch.exp(logits) * logits_mask

	# 	if self.u[index.cpu()].sum() == 0:
	# 		gamma = 1
			
	# 	u = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * exp_logits.sum(1, keepdim=True)
	# 	with torch.no_grad():
	# 		self.u[index.cpu()] = u.cpu()
	# 	prob = torch.exp(logits)/u

	# 	# compute mean of likelihood over positive
	# 	# modified to handle edge cases when there is no positive pair
	# 	# for an anchor point. 
	# 	# Edge case e.g.:- 
	# 	# features of shape: [4,1,...]
	# 	# labels:            [0,1,1,2]
	# 	# loss before mean:  [nan, ..., ..., nan] 
	# 	mask_pos_pairs = mask.sum(1)
	# 	mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
	# 	mean_prob_pos = (mask * prob).sum(1) / mask_pos_pairs

	# 	v = (1 - gamma) * self.v[index.cpu()].cuda() + gamma * mean_prob_pos
	# 	with torch.no_grad():
	# 		self.v[index.cpu()] = v.cpu()

	# 	# loss
	# 	loss = - (self.temperature / self.base_temperature) * torch.log(v)
	# 	loss = loss.view(anchor_count, batch_size).mean()

	# 	return loss

	def forward(self, features, index, labels=None, mask=None, gamma=0.9):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf

		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""
		device = (torch.device('cuda')
				  if features.is_cuda
				  else torch.device('cpu'))
	
		if len(features.shape) < 3:
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
	
		anchor_feature = features[:, 0]
		anchor_count = 1
		
		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
		
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)		
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask

		if self.u[index.cpu()].sum() == 0:
			gamma = 1

		u = (1 - gamma) * (self.u[index.cpu()].cuda() - self.f1_w[index.cpu()].cuda()) + exp_logits.sum(1, keepdim=True)
		with torch.no_grad():
			self.f1_w[index.cpu()] = exp_logits.sum(1, keepdim=True).cpu()
			self.u[index.cpu()] = u.cpu()

		prob = torch.exp(logits)/u

		# compute mean of likelihood over positive
		# modified to handle edge cases when there is no positive pair
		# for an anchor point. 
		# Edge case e.g.:- 
		# features of shape: [4,1,...]
		# labels:            [0,1,1,2]
		# loss before mean:  [nan, ..., ..., nan] 
		mask_pos_pairs = mask.sum(1)
		mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
		mean_prob_pos = (mask * prob).sum(1) / mask_pos_pairs

		# df2 = ((torch.div(prob*logits, self.temperature) - (df1*prob/u))*mask).sum(1) / mask_pos_pairs

		v = (1 - gamma) * (self.v[index.cpu()].cuda() - self.f2_w[index.cpu()].cuda()) + mean_prob_pos
		with torch.no_grad():
			self.f2_w[index.cpu()] = mean_prob_pos.cpu()
			self.v[index.cpu()] = v.cpu()
		
		# loss
		loss = - (self.temperature / self.base_temperature) * torch.log(v)
		loss = loss.view(anchor_count, batch_size).mean()
		
		return loss