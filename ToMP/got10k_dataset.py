import os
import glob
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms 
from utils.heatmap import gaussian_heatmap


class GOT10kDataset(Dataset):
    def __init__(self, root, search_window_size=224, num_frames=3, search_factor = 2, stride=16, sigma=1.0/4.0):
        
        self.target_size = search_window_size / search_factor
        self.search_window_size = search_window_size
        self.num_frames = num_frames
        self.search_factor = search_factor
        self.sigma = sigma
        self.stride = stride
        self.sequences = []

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
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

    def crop_and_resize(self, img, box, crop_center, crop_size, out_size):
        """
        Crop img centered at crop_center with crop_size (h, w), then resize to out_size.
        Adjust box coordinates to the new crop and resize.
        """
        cx, cy = crop_center
        crop_h, crop_w = crop_size

        # Calculate crop coordinates
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x2 = int(round(cx + crop_w / 2.0))
        y2 = int(round(cy + crop_h / 2.0))

        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(0, x2 - img.shape[1])
        y2_pad = max(0, y2 - img.shape[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        patch = img[y1:y2, x1:x2]
        mean_color = [int(np.mean(patch[:, :, i])) for i in range(patch.shape[2])]
        patch = cv2.copyMakeBorder(patch, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=mean_color)
        scale_w = out_size / crop_w
        scale_h = out_size / crop_h
        patch = cv2.resize(patch, (out_size, out_size))
        patch_tensor = self.to_tensor(patch)

        # Adjust bbox for the cropped/resized patch
        dx = x1 - x1_pad
        dy = y1 - y1_pad
        cx_new = (box[0] + box[2] / 2.0 - dx) * scale_w
        cy_new = (box[1] + box[3] / 2.0 - dy) * scale_h
        w_new = box[2] * scale_w
        h_new = box[3] * scale_h
        adj_box = torch.tensor([cx_new - w_new / 2, cy_new - h_new / 2, w_new, h_new], dtype=torch.float32)
        return patch_tensor, adj_box 
    

    def generate_heatmap(self, search_window_size, box, stride):
        heatmap_size = search_window_size // stride
        box = box / stride  # Convert box to heatmap coordinates 
        center =(box[0] + box[2] / 2.0 -0.5,box[1] + box[3] / 2.0-0.5)
        heatmap = gaussian_heatmap(heatmap_size,center, self.sigma)
        return heatmap


    def __getitem__(self, idx):
        """
        Retrieves a sample from the GOT-10k dataset at the given index.

        For the selected sequence, randomly samples a set of frame indices and processes each frame to generate:
        - Centered search window images and bounding boxes, where the target is centered and resized.
        - Noncentered search window images and bounding boxes, where the target is randomly offset within a valid range.
        - Corresponding heatmaps for both centered and noncentered windows.
        
        Returns a dictionary containing lists of images, bounding boxes, heatmaps, sequence name, and frame IDs.
        The boxes are in the format [x,y,w,h]
        
          """
        seq = self.sequences[idx]
        frame_ids = sorted(random.sample(list(seq['valid']), self.num_frames))
        # frame_ids = [0,10]

        centered_images, centered_bboxes, centered_heatmaps = [], [], []
        noncentered_images, noncentered_bboxes, noncentered_heatmaps = [], [], []

        for fid in frame_ids:
            img = cv2.cvtColor(cv2.imread(seq['images'][fid]), cv2.COLOR_BGR2RGB)
            box = seq['gt'][fid]
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            sh = box[3]  * self.search_factor
            sw = box[2] * self.search_factor

            # Template: crop centered on its own bbox
            centered_img, centered_bbox = self.crop_and_resize(
                img, box, (cx, cy), (sh, sw), self.search_window_size
            )
            centered_heatmaps.append(self.generate_heatmap(
                self.search_window_size, centered_bbox, self.stride
            ))

            centered_images.append(centered_img)
            if (centered_bbox < 0).any():
                print(f"Error: Negative values in centered_bbox for sequence {seq['seq_name']}")
            centered_bboxes.append(centered_bbox)

            # Search: random center, but bbox fully inside search window with margin

            # Compute allowed range for center
            margin_h = box[3] 
            margin_w = box[2] 



            min_cx = cx - margin_w/2.0 
            max_cx = cx + margin_w/2.0
            min_cy = cy - margin_h/2.0
            max_cy = cy + margin_h/2.0

            # If the allowed range is invalid (e.g., box too close to edge), fall back to center
            if min_cx >= max_cx or min_cy >= max_cy:
                cx_rand, cy_rand = cx, cy
            else:
                cx_rand = np.random.uniform(min_cx, max_cx)
                cy_rand = np.random.uniform(min_cy, max_cy)

            # cx_rand, cy_rand = cx, cy

            noncentered_img, noncentered_bbox = self.crop_and_resize(
                img, box, (cx_rand, cy_rand), (sh, sw), self.search_window_size
            )
            noncentered_images.append(noncentered_img)
            if (noncentered_bbox < 0).any():
                print(f"Error: Negative values in noncentered_bbox for sequence {seq['seq_name']}")
            noncentered_bboxes.append(noncentered_bbox)

            noncentered_heatmaps.append(self.generate_heatmap(
                self.search_window_size, noncentered_bbox, self.stride
            ))

        return {
            'centered_images': centered_images,         # List of [C, template_size, template_size]
            'centered_bbox': centered_bboxes,             # List of [4]
            'centered_heatmaps': centered_heatmaps,       # List of [C, search_window_size, search_window_size]
            'noncentered_images': noncentered_images,   # List of [C, search_window_size, search_window_size]
            'noncentered_bbox': noncentered_bboxes,       # List of [4]
            'noncentered_heatmaps': noncentered_heatmaps, # List of [C, search_window_size, search_window_size]
            'seq_name': seq['seq_name'],
            'frame_ids': frame_ids
        }

    def upscale_heatmap(self, heatmap):
        """
        Upscale the heatmap by self.stride using nearest neighbor interpolation.
        """
        upscaled_size = (heatmap.shape[0] * self.stride, heatmap.shape[1] * self.stride)
        upscaled_heatmap = cv2.resize(heatmap, upscaled_size, interpolation=cv2.INTER_NEAREST)
        return upscaled_heatmap
    
    def show_sample(self, sample):
        import matplotlib.pyplot as plt
        import numpy as np

        def denormalize(img_tensor):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = (img * std) + mean
            img = np.clip(img, 0, 1)
            return img

        num_frames = len(sample['frame_ids'])

        fig, axs = plt.subplots(num_frames, 4, figsize=(16, 4 * num_frames))
        for i in range(num_frames):
            # Centered
            img_tensor = sample['centered_images'][i]
            bbox = sample['centered_bbox'][i]
            heatmap = sample['centered_heatmaps'][i].cpu().numpy() if hasattr(sample['centered_heatmaps'][i], 'cpu') else sample['centered_heatmaps'][i]
            heatmap = self.upscale_heatmap(heatmap)
            img = denormalize(img_tensor)
            x, y, w, h = bbox.numpy()
            axs[i, 0].imshow(img)
            axs[i, 0].add_patch(plt.Rectangle((x, y), w, h, edgecolor='g', facecolor='none', linewidth=2))
            axs[i, 0].set_title(f"Centered Search {i}")
            axs[i, 0].axis('off')
            axs[i, 1].imshow(heatmap, cmap='hot')
            axs[i, 1].set_title("Centered Heatmap")
            axs[i, 1].axis('off')

            # Noncentered
            img_tensor_nc = sample['noncentered_images'][i]
            bbox_nc = sample['noncentered_bbox'][i]
            heatmap_nc = sample['noncentered_heatmaps'][i].cpu().numpy() if hasattr(sample['noncentered_heatmaps'][i], 'cpu') else sample['noncentered_heatmaps'][i]
            heatmap_nc = self.upscale_heatmap(heatmap_nc)
            img_nc = denormalize(img_tensor_nc)
            x_nc, y_nc, w_nc, h_nc = bbox_nc.numpy()
            axs[i, 2].imshow(img_nc)
            axs[i, 2].add_patch(plt.Rectangle((x_nc, y_nc), w_nc, h_nc, edgecolor='r', facecolor='none', linewidth=2))
            axs[i, 2].set_title(f"Noncentered Search {i}")
            axs[i, 2].axis('off')
            axs[i, 3].imshow(heatmap_nc, cmap='hot')
            axs[i, 3].set_title("Noncentered Heatmap")
            axs[i, 3].axis('off')

        plt.suptitle(f"Seq: {sample['seq_name']}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
