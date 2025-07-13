import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFLayer(nn.Module):
    def __init__(self, num_centers=5, out_channels=1, spatial_size=20, rbf_sigma=0.4):
        super().__init__()
        self.num_centers = num_centers
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.rbf_sigma = rbf_sigma

        self.centers = nn.Parameter(torch.rand(num_centers, 2))  # (num_centers, 2)
        self.weights = nn.Parameter(torch.randn(out_channels, num_centers))

        y, x = torch.meshgrid(
            torch.linspace(0, 1, spatial_size),
            torch.linspace(0, 1, spatial_size),
            indexing="ij"
        )
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(-1, 2))  # [H*W, 2]

    def forward(self):
        dists = torch.cdist(self.grid, self.centers, p=2)  # [H*W, num_centers]
        rbf_vals = torch.exp(- (dists / self.rbf_sigma) ** 2)  # [H*W, num_centers]
        output = torch.matmul(rbf_vals, self.weights.T)  # [H*W, out_channels]
        return output.view(self.out_channels, self.spatial_size, self.spatial_size)  # [C, H, W]

class ResidualFunction(nn.Module):
    def __init__(self, spatial_size=20, rbf_centers=5, rbf_sigma=0.4):
        super().__init__()
        self.spatial_size = spatial_size
        self.vc_rbf = RBFLayer(rbf_centers, 1, spatial_size, rbf_sigma)
        self.mc_rbf = RBFLayer(rbf_centers, 1, spatial_size, rbf_sigma)
        self.yc_rbf = RBFLayer(rbf_centers, 1, spatial_size, rbf_sigma)

    def forward(self, s):
        if s.dim() == 4:
            s = s.squeeze(1)  # [B, H, W]
        B, H, W = s.shape
        vc = self.vc_rbf()        # [1, H, W]
        mc = torch.sigmoid(self.mc_rbf())  # [1, H, W]
        yc = self.yc_rbf()        # [1, H, W]
        residual = vc * (mc * s + (1 - mc) * F.relu(s) - yc)
        return residual
