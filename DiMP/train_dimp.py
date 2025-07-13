import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dimpnet import DiMPNet
from data.got10k_dataset import GOT10kDataset
from utils.heatmap import generate_heatmap
from utils.box_ops import box_cxcywh_to_xyxy
from tqdm import tqdm

VAL_PATH = "/home/artemis/DNN/datasets/GOT-10k/val_data/val"
WEIGHTS_PATH = "/home/artemis/DNN/DiMP/models/dimpnet_got10k.pth"

def train_dimp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiMPNet().to(device)
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    dataset = GOT10kDataset(root=VAL_PATH, size=224, num_frames=2)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.2)

    for epoch in range(10):
        model.train()
        total_cls, total_iou = 0.0, 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            template = batch['images'][0].to(device)
            search   = batch['images'][1].to(device)
            gt_box   = batch['bbox'][1].to(device)

            batch_idx = torch.arange(len(gt_box), device=device).unsqueeze(1).float()
            boxes_xyxy = box_cxcywh_to_xyxy(gt_box)
            iou_boxes = torch.cat([batch_idx, boxes_xyxy], dim=1)

            score_map,pred_iou = model(template, search, iters=5, return_bbox=True, candidate_boxes=iou_boxes)
            res_loss = model.compute_residual_loss(score_map)

            gt_iou = torch.ones_like(pred_iou)
            iou_loss = mse(pred_iou, gt_iou)

            loss = res_loss + 1.0 * iou_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_cls += res_loss.item()
            total_iou += iou_loss.item()
        print(f"Epoch {epoch+1} - Cls: {total_cls:.4f}, IoU: {total_iou:.4f}")
    torch.save(model.state_dict(), WEIGHTS_PATH)

if __name__ == "__main__":
    train_dimp()