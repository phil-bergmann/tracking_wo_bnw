# https://matplotlib.org/cycler/


from .config import cfg, get_output_dir


import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import io
from PIL import Image
from scipy.optimize import linear_sum_assignment
from cycler import cycler as cy
from collections import defaultdict

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# get all colors with
#colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)
colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']


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
	length = ind0.size()[0]
	loss = nn.CrossEntropyLoss()

	"""NOT NEEDED ANYMORE, use cross entropy loss
	# Build targets for softmax output
	#tar0 = Variable(torch.zeros((length,300))).cuda()
	#tar1 = Variable(torch.zeros((length,300))).cuda()
	#for i,j in enumerate(ind0.data.cpu().numpy()):
		tar0[i,j] = 1
	for i,j in enumerate(ind1.data.cpu().numpy()):
		tar1[i,j] = 1
	"""



	# Compare the output vectors
	cost0 = Variable(torch.zeros((length,length)).cuda())
	cost1 = Variable(torch.zeros((length,length)).cuda())
	for i in range(length):
		for j in range(length):
			#cost0[i,j] = F.pairwise_distance(bbox0[i].view(1,-1), tar0[j].view(1,-1))
			#cost1[i,j] = F.pairwise_distance(bbox1[i].view(1,-1), tar1[j].view(1,-1))
			cost0[i,j] = loss(bbox0[i].view(1,-1), ind0[j])
			cost1[i,j] = loss(bbox1[i].view(1,-1), ind1[j])

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
		#im = buf.getvalue()
		buf.close()
		return im

def plot_sequence(tracks, db, output_dir):
	"""Plots a whole sequence

	Args:
		tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
		db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
		output_dir (String): Directory where to save the resultind images
	"""

	print("[*] Plotting whole sequence to {}".format(output_dir))

	if not osp.exists(output_dir):
		os.makedirs(output_dir)

	# infinte color loop
	cyl = cy('ec', colors)
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	for i,v in enumerate(db):
		im_path = v['im_path']
		im_name = osp.basename(im_path)
		im_output = osp.join(output_dir, im_name)
		im = cv2.imread(im_path)
		im = im[:, :, (2, 1, 0)]

		fig, ax = plt.subplots(1,1)
		ax.imshow(im, aspect='equal')

		for j,t in tracks.items():
			if i in t.keys():
				t_i = t[i]
				ax.add_patch(
				plt.Rectangle((t_i[0], t_i[1]),
					  t_i[2] - t_i[0],
					  t_i[3] - t_i[1], fill=False,
					  linewidth=1.0, **styles[j])
				)

		plt.axis('off')
		plt.tight_layout()
		plt.draw()
		plt.savefig(im_output)
		plt.close()


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
	#output_dir = get_output_dir("anchor_gt_demo")
	im_paths = blobs['im_paths']
	if not name:
		im0_name = osp.basename(im_paths[0])
	else:
		im0_name = str(name)+".jpg"
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = blobs['im_info'][0,2]

	tracks = tracks.data.cpu().numpy() / im_scales
	num_tracks = tracks.shape[0]

	#print(tracks.shape)
	#print(tracks)

	fig, ax = plt.subplots(1,2,figsize=(12, 6))

	ax[0].imshow(im0, aspect='equal')
	ax[1].imshow(im1, aspect='equal')

	# infinte color loop
	cyl = cy('ec', colors)
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

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
	image = None
	if output_dir:
		im_output = osp.join(output_dir,im0_name)
		plt.savefig(im_output)
	else:
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	plt.close()
	return image


def plot_tracks_video(blobs, tracks, output_dir):
	#output_dir = get_output_dir("anchor_gt_demo")

	if not osp.exists(output_dir):
		os.makedirs(output_dir)

	im_paths = blobs['im_paths']
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = blobs['im_info'][0,2]

	tracks = tracks.data.cpu().numpy() / im_scales
	num_tracks = tracks.shape[0]

	#print(tracks.shape)
	#print(tracks)
	for i in range(1,num_tracks+1):

		fig, ax = plt.subplots(1,2,figsize=(12, 6))

		ax[0].imshow(im0, aspect='equal')
		ax[1].imshow(im1, aspect='equal')

		# infinte color loop
		cyl = cy('ec', colors)
		loop_cy_iter = cyl()
		styles = defaultdict(lambda : next(loop_cy_iter))

		ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

		for j in range(i):
			t = tracks[j]
			t0 = t[0]
			t1 = t[1]
			ax[0].add_patch(
				plt.Rectangle((t0[0], t0[1]),
						  t0[2] - t0[0],
						  t0[3] - t0[1], fill=False,
						  linewidth=1.0, **styles[j])
				)
			ax[1].add_patch(
				plt.Rectangle((t1[0], t1[1]),
						  t1[2] - t1[0],
						  t1[3] - t1[1], fill=False,
						  linewidth=1.0, **styles[j])
				)

		plt.axis('off')
		plt.tight_layout()
		plt.draw()
		im_output = osp.join(output_dir,"{:03d}.jpg".format(i))
		plt.savefig(im_output)
		plt.close()


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
	plt.close()

def plot_simple(mb, bb0, bb1, output_dir):
	im_paths = mb['im_paths']
	im0_name = osp.basename(im_paths[0])
	im_output = osp.join(output_dir,im0_name)
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	bb0 = bb0[:,0:4]
	bb1 = bb1[:,0:4]

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

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.savefig(im_output)

def boxes2rois(boxes, cl=1):
	rois_score = boxes.new(boxes.size()[0],1).zero_()
	rois_bb = boxes[:, cl*4:(cl+1)*4]
	rois = torch.cat((rois_score, rois_bb),1)

	return rois