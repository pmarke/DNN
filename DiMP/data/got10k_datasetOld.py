import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dimp.utils.heatmap import generate_heatmap

class GOT10kDataset(Dataset):
    def __init__(self, root, size=256, sigma=3, heatmap_size=20, use_labels=True):
        self.size = size
        self.sigma = sigma
        self.heatmap_size = heatmap_size
        self.use_labels = use_labels
        self.sequences = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        for seq_dir in sorted(glob.glob(os.path.join(root, '*'))):
            images = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
            gt_path = os.path.join(seq_dir, 'groundtruth.txt')
            if len(images) < 2 or not os.path.isfile(gt_path):
                continue

            gt = np.loadtxt(gt_path, delimiter=',')
            valid = np.arange(len(gt))

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

        x, y, w, h = gt[idx2]
        cx = (x + w / 2) * self.heatmap_size / img2.shape[1]
        cy = (y + h / 2) * self.heatmap_size / img2.shape[0]
        box = torch.tensor([cx, cy, w * self.heatmap_size / img2.shape[1], h * self.heatmap_size / img2.shape[0]])
        return tmpl, srch, box
