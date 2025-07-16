import torch
import numpy as np
import cv2

def gaussian_heatmap(size, center, sigma=3):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0, y0 = center
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def generate_label(size=256, sigma=3):
    heatmap = gaussian_heatmap(size, (size//2, size//2), sigma)
    return torch.tensor(heatmap).unsqueeze(0).float()  # 1 x H x W

def calculate_iou(boxes1, boxes2, eps=1e-6):
    """
    Calculates IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor[N, 4] in (x1, y1, x2, y2) format
        boxes2: Tensor[N, 4] in (x1, y1, x2, y2) format
        eps: Small value to avoid division by zero

    Returns:
        ious: Tensor[N] IoU values between corresponding boxes
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1 + area2 - inter_area + eps

    return inter_area / union_area

def box_xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1], dim=-1)

def box_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2).
    
    Args:
        boxes (Tensor): shape [..., 4] with (cx, cy, w, h)

    Returns:
        Tensor: same shape [..., 4] with (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)