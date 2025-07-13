import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.pyplot as plt

class GOT10kDataset(Dataset):
    def __init__(self, root, size=224, num_frames=2):
        self.size = size
        self.num_frames = num_frames
        self.sequences = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        for seq_dir in sorted(glob.glob(os.path.join(root, '*'))):
            images = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
            gt_path = os.path.join(seq_dir, 'groundtruth.txt')
            if len(images) < num_frames or not os.path.isfile(gt_path):
                continue

            gt = np.loadtxt(gt_path, delimiter=',')
            valid = np.arange(len(gt))
            seq_name = os.path.basename(seq_dir)

            if len(valid) >= num_frames:
                self.sequences.append({
                    'images': images,
                    'gt': gt,
                    'valid': valid,
                    'seq_name': seq_name
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        frame_ids = sorted(random.sample(list(seq['valid']), self.num_frames))
        images = []
        bboxes = []
        valid = []

        for fid in frame_ids:
            img = cv2.cvtColor(cv2.imread(seq['images'][fid]), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            images.append(self.transform(img))  # [C, size, size]

            # Resize bbox from original image size to new size
            x, y, w, h = seq['gt'][fid]
            scale_x = self.size / orig_w
            scale_y = self.size / orig_h
            x_resized = x * scale_x
            y_resized = y * scale_y
            w_resized = w * scale_x
            h_resized = h * scale_y
            bboxes.append(torch.tensor([x_resized, y_resized, w_resized, h_resized], dtype=torch.float32))
            valid.append(True)

        return {
            'images': images,           # List of tensors [C, H, W]
            'bbox': bboxes,             # List of [x, y, w, h] tensors (resized)
            'valid': valid,             # List of bools
            'seq_name': seq['seq_name'],
            'frame_ids': frame_ids
        }

    def show_sample(self, sample):
        for i, (img_tensor, bbox) in enumerate(zip(sample['images'], sample['bbox'])):
            img = img_tensor.permute(1, 2, 0).numpy()
            x, y, w, h = bbox.numpy()
            plt.figure()
            plt.imshow(img)
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=2))
            plt.title(f"Seq: {sample['seq_name']} Frame: {sample['frame_ids'][i]}")
            plt.axis('off')
        plt.show()