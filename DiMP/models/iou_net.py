# Placeholder - content will be inserted later
import torch
import torch.nn as nn
from torchvision.ops import roi_align

class IoUNet(nn.Module):
    def __init__(self, in_channels=64, pool_size=7, fc_dim=256):
        super().__init__()
        self.pool_size = pool_size
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * pool_size * pool_size, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_map, boxes, spatial_scale=1.0):
        roi_feats = roi_align(feat_map, boxes, output_size=self.pool_size, spatial_scale=spatial_scale)
        x = self.roi_conv(roi_feats)
        x = x.flatten(start_dim=1)
        return self.fc(x)
