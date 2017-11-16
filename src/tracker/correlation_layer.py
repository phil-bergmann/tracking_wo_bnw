import torch

def correlation_layer(im0_fc7, im1_fc7):
	"""Calculate correlations between region proposals

	Expects two tensors/variables containing the fc7 features of the proposals, size 300x4096,
	and calculates their correlation (Scalar Prouct between every feature vector). Output thus
	is a Matrix with size 300x300

	torch.Size([300, 4096])
	torch.Size([300, 5])

	For reference see Convolutional neural network architecture for geometric matching, Rocco et al.
	"""
	# Normalize every 4096 vector with L2 norm
	norm0 = torch.norm(im0_fc7,2,dim=1,keepdim=True)
	norm1 = torch.norm(im1_fc7,2,dim=1,keepdim=True)

	im0_norm = im0_fc7.div(norm0)
	im1_norm = im1_fc7.div(norm1)
	
	return torch.mm(im0_norm,im1_norm.t())