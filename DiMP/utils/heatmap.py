# Placeholder - content will be inserted later
import torch
import numpy as np

def gaussian_heatmap(size, center, sigma=3, device='cpu', dtype=torch.float32):
    x = torch.arange(0, size, dtype=dtype, device=device)
    y = x.unsqueeze(1)
    cx, cy = center
    cx = torch.tensor(cx, dtype=dtype, device=device)
    cy = torch.tensor(cy, dtype=dtype, device=device)
    return torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def generate_heatmap(boxes, size=20, sigma=2):
    batch = []
    for box in boxes:
        cx, cy = box[0], box[1]
        heat = gaussian_heatmap(size, (cx, cy), sigma)
        batch.append(torch.tensor(heat).unsqueeze(0))
    return torch.stack(batch)
