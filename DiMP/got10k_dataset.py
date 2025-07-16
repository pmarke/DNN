import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random

import matplotlib.pyplot as plt

def gaussian_heatmap(size, center, sigma=3):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    cx, cy = center
    heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]

class GOT10kDataset(Dataset):
    def __init__(self, root, size=256, use_labels=True, sigma=10):
        self.size = size
        self.use_labels = use_labels
        self.sequences = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.sigma = sigma

        for seq_dir in sorted(glob.glob(os.path.join(root, '*'))):
            images = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
            gt_path = os.path.join(seq_dir, 'groundtruth.txt')

            if len(images) < 2 or not os.path.isfile(gt_path):
                continue

            gt = np.loadtxt(gt_path, delimiter=',')
            num_frames = len(gt)

            if use_labels:
                absence_path = os.path.join(seq_dir, 'absence.label')
                cut_path = os.path.join(seq_dir, 'cut_by_image.label')
                absence = np.loadtxt(absence_path, dtype=np.int32) if os.path.isfile(absence_path) else np.zeros(num_frames, dtype=np.int32)
                cut = np.loadtxt(cut_path, dtype=np.int32) if os.path.isfile(cut_path) else np.zeros(num_frames, dtype=np.int32)
                valid = np.where((absence == 0) & (cut == 0))[0]
            else:
                valid = np.arange(num_frames)

            if len(valid) >= 2:
                self.sequences.append((images, gt, valid))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        images, gt, valid = self.sequences[idx]
        idx1, idx2 = sorted(random.sample(list(valid), 2))

        img1 = cv2.cvtColor(cv2.imread(images[idx1]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(images[idx2]), cv2.COLOR_BGR2RGB)

        tmpl = self.transform(img1)
        srch = self.transform(img2)

        # Use frame 2 groundtruth to build target heatmap
        x, y, w, h = gt[idx2]
        cx = int((x + w / 2) * self.size / img2.shape[1])
        cy = int((y + h / 2) * self.size / img2.shape[0])
        label = gaussian_heatmap(self.size, (cx, cy), sigma=self.sigma)

        return tmpl, srch, label

    def show_sample(self, idx=None):
        """Displays a sample template, search, and label heatmap."""
        if idx is None:
            idx = random.randint(0, len(self) - 1)
        tmpl, srch, label = self[idx]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(tmpl.permute(1, 2, 0).cpu())
        axs[0].set_title("Template")
        axs[1].imshow(srch.permute(1, 2, 0).cpu())
        axs[1].set_title("Search")
        axs[2].imshow(label.squeeze(0).cpu(), cmap="hot")
        axs[2].set_title("Gaussian Label")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
