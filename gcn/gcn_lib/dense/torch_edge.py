import torch
from torch import nn


def _pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def _knn_matrix(x, k=16, self_loop=True):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    x = x.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    if self_loop:
        _, nn_idx = torch.topk(-_pairwise_distance(x.detach()), k=k)
    else:
        _, nn_idx = torch.topk(-_pairwise_distance(x.detach()), k=k+1)
        nn_idx = nn_idx[:, :, 1:]
    center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class Dilated2d(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated2d, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            raise NotImplementedError('stochastic currently is not supported')
            # if torch.rand(1) < self.epsilon and self.training:
            #     num = self.k * self.dilation
            #     randnum = torch.randperm(num)[:self.k]
            #     edge_index = edge_index[:, :, :, randnum]
            # else:
            #     edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DilatedKnn2d(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, self_loop=True, stochastic=False, epsilon=0.0):
        super(DilatedKnn2d, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self.self_loop = self_loop
        self._dilated = Dilated2d(k, dilation, stochastic, epsilon)
        self.knn = _knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k * self.dilation, self.self_loop)
        return self._dilated(edge_index)


def remove_self_loops(edge_index):
    if edge_index[0, 0, 0, 0] == 0:
        edge_index = edge_index[:, :, :, 1:]
    return edge_index


def add_self_loops(edge_index):
    if edge_index[0, 0, 0, 0] != 0:
        self_loops = torch.arange(0, edge_index.shape[2]).repeat(2, edge_index.shape[1], 1).unsqueeze(-1)
        edge_index = torch.cat((self_loops.to(edge_index.device), edge_index[:, :, :, 1:]), dim=-1)
    return edge_index
