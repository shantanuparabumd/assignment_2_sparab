import torch
import torch.nn as nn
from pytorch3d import loss

from pytorch3d.ops import knn_points, knn_gather

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	
	lossfn = nn.BCEWithLogitsLoss()

	loss=lossfn(voxel_src,voxel_tgt)

	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	
		knn_src = knn_points(point_cloud_tgt, point_cloud_src, K=1)
		knn_tgt = knn_points(point_cloud_src, point_cloud_tgt, K=1)

		# Gather the k-nearest neighbors
		knn_src_points = knn_gather(point_cloud_src, knn_src.idx)
		knn_tgt_points = knn_gather(point_cloud_tgt, knn_tgt.idx)

		# Compute the pairwise distances between the points
		dist_src_to_tgt = torch.norm(point_cloud_src.unsqueeze(2) - knn_tgt_points, dim=-1)
		dist_tgt_to_src = torch.norm(point_cloud_tgt.unsqueeze(2) - knn_src_points, dim=-1)

		# Compute the Chamfer loss
		loss_chamfer = torch.mean(torch.min(dist_src_to_tgt, dim=2).values) + torch.mean(torch.min(dist_tgt_to_src, dim=2).values)
		
		return loss_chamfer

def smoothness_loss(mesh_src):

		loss_laplacian = loss.mesh_laplacian_smoothing(mesh_src,method='uniform')
		# implement laplacian smoothening loss
		return loss_laplacian