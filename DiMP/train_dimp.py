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

torch.set_printoptions(linewidth=500)
torch.set_printoptions(precision=3)

def train_dimp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiMPNet().to(device)
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    dataset = GOT10kDataset(root=VAL_PATH, template_size=256, num_frames=2)
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.2)

    # sample = dataset[0] 
    # dataset.show_sample(sample)

    for epoch in range(10):
        model.train()
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            total_cls, total_iou = 0.0, 0.0
            # Each batch entry is a list of length batch_size, each containing a list of length num_frames
            # We use the first frame as template, second as search (adjust if num_frames > 2)
            # Collate lists into tensors
            # print(batch['seq_name'])
            # print(batch['search_bboxes'])
            # print(batch['search_bboxes'][0])
            # print(batch.keys())
            
            template = batch['template_images'][0].to(device)  # [C, H, W]
            search = batch['search_images'][1].to(device)      # [C, H, W]
            gt_box = batch['search_bboxes'][1].to(device)      # [4]

            # sample = dataset[0]
            # print(sample['seq_name'])
            # print(sample['frame_ids'])
            # template = sample['template_images'][0].to(device).unsqueeze(0)  # [C, H, W]
            # search = sample['search_images'][1].to(device).unsqueeze(0)      # [C, H, W]
            # gt_box = sample['search_bboxes'][1].to(device).unsqueeze(0)      # [4]
            # print("Template shape:", template.shape)

            batch_idx = torch.arange(len(gt_box), device=device).unsqueeze(1).float()
            boxes_xyxy = box_cxcywh_to_xyxy(gt_box)
            iou_boxes = torch.cat([batch_idx, boxes_xyxy], dim=1)

            # score_map, pred_iou = model(template, search, iters=5, return_bbox=True, candidate_boxes=iou_boxes)
            score_map = model(template, search, iters=5, return_bbox=False, candidate_boxes=iou_boxes)
            res_loss = model.compute_residual_loss(score_map, bbox=gt_box, search_img_size=search.shape[-1])
            # print(score_map[0])

            # gt_iou = torch.ones_like(pred_iou)
            # iou_loss = mse(pred_iou, gt_iou)

            # loss = res_loss + 1.0 * iou_loss
            loss = res_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_cls += res_loss.item()
            # total_iou += iou_loss.item()
        print(f"Epoch {epoch+1} - Cls: {total_cls:.4f}, IoU: {total_iou:.4f}")
    torch.save(model.state_dict(), WEIGHTS_PATH)

if __name__ == "__main__":
    train_dimp()