# Placeholder - content will be inserted later
import torch
import numpy as np

def gaussian_heatmap(size, center, sigma=3):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    cx, cy = center
    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def generate_heatmap(boxes, size=20, sigma=2):
    batch = []
    for box in boxes:
        cx, cy = box[0], box[1]
        heat = gaussian_heatmap(size, (cx, cy), sigma)
        batch.append(torch.tensor(heat).unsqueeze(0))
    return torch.stack(batch)
