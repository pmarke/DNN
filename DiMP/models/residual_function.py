
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TriangularBasisMapGenerator(nn.Module):
    def __init__(self, num_bases=50, out_channels=1, spatial_size=14, max_distance=None):
        super().__init__()
        self.num_bases = num_bases
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.max_distance = max_distance or math.sqrt(2) * spatial_size

        # Basis centers r_k
        self.register_buffer("r_centers", torch.linspace(0, self.max_distance, steps=num_bases))
        self.delta = self.r_centers[1] - self.r_centers[0]
        # self.delta = 0.1

        # Learnable coefficients phi_k: [out_channels, num_bases]
        self.phi = nn.Parameter(torch.zeros(out_channels, num_bases))

        # Precompute [H, W, 2] grid
        y, x = torch.meshgrid(
            torch.arange(spatial_size), torch.arange(spatial_size), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([x, y], dim=-1).float())  # [H, W, 2]

    def forward(self, center):
        """
        Args:
            center: tensor [2] in grid space (cx, cy)
        Returns:
            output: [out_channels, H, W]
        """
        cx, cy = center[0], center[1]
        dist = torch.norm(self.grid - torch.tensor([cx, cy], device=self.grid.device), dim=-1)  # [H, W]

        # Compute basis stack: [K, H, W]
        basis_stack = torch.stack([
            (1.0 - torch.abs(dist - rk) / self.delta).clamp(min=0)
            for rk in self.r_centers
        ], dim=0)

        # Weighted sum: [out_channels, H, W]
        return torch.einsum("oc,chw->ohw", self.phi, basis_stack)



class ResidualFunction(nn.Module):
    def __init__(self, spatial_size=14, rbf_centers=50):
        super().__init__()
        self.spatial_size = spatial_size
        self.vc_map = TriangularBasisMapGenerator(num_bases=rbf_centers, out_channels=1, spatial_size=spatial_size)
        self.mc_map = TriangularBasisMapGenerator(num_bases=rbf_centers, out_channels=1, spatial_size=spatial_size)
        self.yc_map = TriangularBasisMapGenerator(num_bases=rbf_centers, out_channels=1, spatial_size=spatial_size)

    def forward(self, s, center):
        """
        Args:
            s: score map, shape [B, 1, H, W] or [B, H, W]
            center: Tensor [B, 2] of ground truth center in score map space
        Returns:
            residual: [B, H, W]
        """
        if s.dim() == 4:
            s = s.squeeze(1)  # [B, H, W]

        B, H, W = s.shape
        assert H == self.spatial_size and W == self.spatial_size

        residuals = []
        for i in range(B):
            vc = F.softplus(self.vc_map(center[i]))[0]          # [H, W], ≥ 0
            mc = torch.sigmoid(self.mc_map(center[i]))[0]       # [H, W], ∈ [0, 1]
            yc = torch.tanh(self.yc_map(center[i]))[0]          # [H, W], ∈ [-1, 1]
            si = s[i]                                            # [H, W]

            ri = vc * (mc * si + (1 - mc) * F.relu(si) - yc)
            residuals.append(ri)

        return torch.stack(residuals, dim=0)  # [B, H, W]

