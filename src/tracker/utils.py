# https://matplotlib.org/cycler/


from .config import cfg, get_output_dir


import matplotlib.pyplot as plt
import numpy as np
from os import path as osp
import cv2
import io
from PIL import Image
from scipy.optimize import linear_sum_assignment
from cycler import cycler as cy
from collections import defaultdict

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def hungarian_iou(pred_tracks, gt_tracks):
	"""Returns the best matching of the tracks
	
	pred_tracks, gt_tracks: torch variable (N,1,4), (number_of_tracks,image_number,coordinates)
	"""

	# Calculate IoU Matrix, sum of IoU of bboxes in first image and bboxes in second image
	IoU0 = bbox_overlaps(pred_tracks[:,0,:], gt_tracks[:,0,:])
	IoU1 = bbox_overlaps(pred_tracks[:,1,:], gt_tracks[:,1,:])

	IoU = IoU1 + IoU2

	# hungarian optimizes for minimal cost, we want max. keep it positive for hung to work
	cost = 2 - IoU

	row_ind, col_ind = linear_sum_assignment(cost.data.cpu().numpy())

	# Probably better LongTensor as this will be used for indexing
	row_ind = Variable(torch.from_numpy(row_ind), require_grad=False).long().cuda()
	col_ind = Variable(torch.from_numpy(col_ind), require_grad=False).long().cuda()

	return row_ind, col_ind, IoU

def hungarian_soft(bbox0, bbox1, ind0, ind1):
	"""Returns targets that are assigned with hungarian for best matching

	bbox: softmax out of the rnn (N,300)
	ind: Variable size (N), with the indexes of the anchors matching best to gt
	"""
	# Build targets for softmax output
	length = ind0.size()[0]
	tar0 = Variable(torch.zeros((length,300))).cuda()
	tar1 = Variable(torch.zeros((length,300))).cuda()
	for i,j in enumerate(ind0.data.cpu().numpy()):
		tar0[i,j] = 1
	for i,j in enumerate(ind1.data.cpu().numpy()):
		tar1[i,j] = 1

	# Compare the output vectors
	cost0 = Variable(torch.zeros((length,length)).cuda())
	cost1 = Variable(torch.zeros((length,length)).cuda())
	for i in range(length):
		for j in range(length):
			cost0[i,j] = F.pairwise_distance(bbox0[i].view(1,-1), tar0[j].view(1,-1))
			cost1[i,j] = F.pairwise_distance(bbox1[i].view(1,-1), tar1[j].view(1,-1))

	cost = cost0 + cost1

	# row_ind on scare matrix sorted as arange
	row_ind, col_ind = linear_sum_assignment(cost.data.cpu().numpy())

	col_ind = Variable(torch.from_numpy(col_ind)).cuda()

	return ind0[col_ind], ind1[col_ind]

# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
	"""
	Parameters
	----------
	boxes: (N, 4) ndarray or tensor or variable
	query_boxes: (K, 4) ndarray or tensor or variable
	Returns
	-------
	overlaps: (N, K) overlap between boxes and query_boxes
	"""
	if isinstance(boxes, np.ndarray):
		boxes = torch.from_numpy(boxes)
		query_boxes = torch.from_numpy(query_boxes)
		out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
	else:
		out_fn = lambda x: x

	box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
			(boxes[:, 3] - boxes[:, 1] + 1)
	query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
			(query_boxes[:, 3] - query_boxes[:, 1] + 1)

	iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
	ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
	ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
	overlaps = iw * ih / ua
	return out_fn(overlaps)

def plot_bb(mb, bb0, bb1, gt_tracks, output_dir=None):
	#output_dir = get_output_dir("anchor_gt_demo")
	im_paths = mb['im_paths']
	im0_name = osp.basename(im_paths[0])
	im_output = osp.join(output_dir,im0_name)
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = mb['blobs']['im_info'][0,2]

	bb0 = bb0.data.cpu().numpy() / im_scales
	bb1 = bb1.data.cpu().numpy() / im_scales

	fig, ax = plt.subplots(1,2,figsize=(12, 12))

	ax[0].imshow(im0, aspect='equal')
	ax[1].imshow(im1, aspect='equal')

	for bb in bb0:
		ax[0].add_patch(
			plt.Rectangle((bb[0], bb[1]),
					  bb[2] - bb[0],
					  bb[3] - bb[1], fill=False,
					  edgecolor='red', linewidth=1.0)
			)

	for bb in bb1:
		ax[1].add_patch(
			plt.Rectangle((bb[0], bb[1]),
					  bb[2] - bb[0],
					  bb[3] - bb[1], fill=False,
					  edgecolor='red', linewidth=1.0)
			)

	for gt in gt_tracks:
		for i in range(2):
			ax[i].add_patch(
			plt.Rectangle((gt[i][0], gt[i][1]),
					  gt[i][2] - gt[i][0],
					  gt[i][3] - gt[i][1], fill=False,
					  edgecolor='blue', linewidth=1.0)
			)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	if output_dir:
		plt.savefig(im_output)
	else:
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		im = Image.open(buf)
		buf.close()
		return im

def plot_tracks(mb, tracks, gt_tracks=None, output_dir=None, name=None):
	#output_dir = get_output_dir("anchor_gt_demo")
	im_paths = mb['im_paths']
	if not name:
		im0_name = osp.basename(im_paths[0])
	else:
		im0_name = str(name)+".jpg"
	im_output = osp.join(output_dir,im0_name)
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = mb['blobs']['im_info'][0,2]

	tracks = tracks.data.cpu().numpy() / im_scales

	print(tracks.shape)
	print(tracks)

	fig, ax = plt.subplots(1,2,figsize=(12, 12))

	ax[0].imshow(im0, aspect='equal')
	ax[1].imshow(im1, aspect='equal')

	# infinte color loop
	cyl = cy('ec', 'rgbymcwk')
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	for i,t in enumerate(tracks):
		t0 = t[0]
		t1 = t[1]
		ax[0].add_patch(
			plt.Rectangle((t0[0], t0[1]),
					  t0[2] - t0[0],
					  t0[3] - t0[1], fill=False,
					  linewidth=1.0, **styles[i])
			)
		ax[1].add_patch(
			plt.Rectangle((t1[0], t1[1]),
					  t1[2] - t1[0],
					  t1[3] - t1[1], fill=False,
					  linewidth=1.0, **styles[i])
			)

	if gt_tracks:
		for gt in gt_tracks:
			for i in range(2):
				ax[i].add_patch(
				plt.Rectangle((gt[i][0], gt[i][1]),
					  gt[i][2] - gt[i][0],
					  gt[i][3] - gt[i][1], fill=False,
					  edgecolor='blue', linewidth=1.0)
				)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	if output_dir:
		plt.savefig(im_output)
	else:
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		im = Image.open(buf)
		buf.close()
		return im


def plot_correlation(im_paths, im_info0, im_info1, cor, rois0, rois1):
	"""Draw correlations."""
	output_dir = get_output_dir("correlation_demo")
	im0_name = osp.basename(im_paths[0])
	im_output = osp.join(output_dir,im0_name)
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]
	

	im_scales0 = im_info0[0,2]
	im_scales1 = im_info1[0,2]

	rois0 = rois0.data.cpu().numpy() / im_scales0
	rois1 = rois1.data.cpu().numpy() / im_scales1
	cor = cor.data.cpu().numpy()
	#print(cor.shape)
	
	# randomly select 5 rois of the first image
	indexes = np.random.random_integers(0,299,5)

	fig, axes = plt.subplots(5,4,figsize=(48, 48))
	for i,ax in enumerate(axes):
		ax[0].imshow(im0, aspect='equal')
		ax[1].imshow(im1, aspect='equal')
		ax[2].imshow(im1, aspect='equal')
		ax[3].imshow(im1, aspect='equal')


		roi0 = rois0[indexes[i],1:]

		scores = cor[indexes[i]]
		order = np.argsort(scores)[::-1]
		
		ax[0].add_patch(
			plt.Rectangle((roi0[0], roi0[1]),
					  roi0[2] - roi0[0],
					  roi0[3] - roi0[1], fill=False,
					  edgecolor='red', linewidth=1.0)
			)

		for j in range(30):
			roi1 = rois1[order[j]][1:]
			score = cor[indexes[i],order[j]]


			ax[j//10+1].add_patch(
				plt.Rectangle((roi1[0], roi1[1]),
				  roi1[2] - roi1[0],
				  roi1[3] - roi1[1], fill=False,
				  edgecolor='red', linewidth=1.0)
			)

			ax[j//10+1].text(roi1[0], roi1[1] - 2,
				'{:.3f}'.format(score),
				#bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=8, color='white')

	#ax.set_title(('{} detections with '
	#		  'p({} | box) >= {:.2f}').format(class_name, class_name,
	#										  thresh),
	#		  fontsize=14)
	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.savefig(im_output)