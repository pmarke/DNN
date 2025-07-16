import os
import glob
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class GOT10kDataset(Dataset):
    def __init__(self, root, template_size=256, num_frames=4, search_factor=2):
        self.template_size = template_size
        self.search_window_size = template_size*search_factor
        self.num_frames = num_frames
        self.search_factor = search_factor
        self.sequences = []
        self.crop_size_scale = 1.2

        self.to_tensor = transforms.ToTensor()

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
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))
        x2 = int(round(cx + crop_w / 2))
        y2 = int(round(cy + crop_h / 2))

        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(0, x2 - img.shape[1])
        y2_pad = max(0, y2 - img.shape[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        patch = img[y1:y2, x1:x2]
        patch = cv2.copyMakeBorder(patch, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
        scale_w = out_size / crop_w
        scale_h = out_size / crop_h
        patch = cv2.resize(patch, (out_size, out_size))
        patch_tensor = self.to_tensor(patch)

        # Adjust bbox for the cropped/resized patch
        dx = x1 - x1_pad
        dy = y1 - y1_pad
        cx_new = (box[0] + box[2] / 2 - dx) * scale_w
        cy_new = (box[1] + box[3] / 2 - dy) * scale_h
        w_new = box[2] * scale_w
        h_new = box[3] * scale_h
        adj_box = torch.tensor([cx_new - w_new / 2, cy_new - h_new / 2, w_new, h_new], dtype=torch.float32)
        print(adj_box)
        return patch_tensor, adj_box

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        frame_ids = sorted(random.sample(list(seq['valid']), self.num_frames))

        template_images, template_bboxes = [], []
        search_images, search_bboxes = [], []

        for fid in frame_ids:
            img = cv2.cvtColor(cv2.imread(seq['images'][fid]), cv2.COLOR_BGR2RGB)
            box = seq['gt'][fid]
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
            th, tw = box[3]*self.crop_size_scale, box[2]*self.crop_size_scale

            # Template: crop centered on its own bbox
            template_img, template_bbox = self.crop_and_resize(
                img, box, (cx, cy), (th, tw), self.template_size
            )
            template_images.append(template_img)
            template_bboxes.append(template_bbox)

            # Search: random center, but bbox fully inside search window with margin
            sh = box[3]  * self.crop_size_scale
            sw = box[2] * self.crop_size_scale

            # still not right

            # Compute allowed range for center
            margin_h = box[3] * self.crop_size_scale
            margin_w = box[2] * self.crop_size_scale
            min_cx = box[0] + margin_w / 2
            max_cx = box[0] + box[2] - margin_w / 2
            min_cx = max(min_cx, sw / 2)
            max_cx = min(max_cx, img.shape[1] - sw / 2)
            min_cy = box[1] + margin_h / 2
            max_cy = box[1] + box[3] - margin_h / 2
            min_cy = max(min_cy, sh / 2)
            max_cy = min(max_cy, img.shape[0] - sh / 2)

            # If the allowed range is invalid (e.g., box too close to edge), fall back to center
            if min_cx >= max_cx or min_cy >= max_cy:
                cx_rand, cy_rand = cx, cy
            else:
                cx_rand = np.random.uniform(min_cx, max_cx)
                cy_rand = np.random.uniform(min_cy, max_cy)

            search_img, search_bbox = self.crop_and_resize(
                img, box, (cx_rand, cy_rand), (sh, sw), self.search_window_size
            )
            search_images.append(search_img)
            search_bboxes.append(search_bbox)

        return {
            'template_images': template_images,      # List of [C, template_size, template_size]
            'template_bboxes': template_bboxes,      # List of [4]
            'search_images': search_images,          # List of [C, search_window_size, search_window_size]
            'search_bboxes': search_bboxes,          # List of [4]
            'seq_name': seq['seq_name'],
            'frame_ids': frame_ids
        }

    def show_sample(self, sample):
        import matplotlib.pyplot as plt

        num_frames = len(sample['frame_ids'])

        # Show template images with template bbox
        for i in range(num_frames):
            img_tensor = sample['template_images'][i]
            bbox = sample['template_bboxes'][i]
            img = img_tensor.permute(1, 2, 0).numpy()
            x, y, w, h = bbox.numpy()
            plt.figure()
            plt.imshow(img)
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=2))
            plt.title(f"Template {i} | Seq: {sample['seq_name']} Frame: {sample['frame_ids'][i]}")
            plt.axis('off')

        # Show search images with template bbox (red) and search bbox (green)
        for i in range(num_frames):
            img_tensor = sample['search_images'][i]
            template_bbox = sample['template_bboxes'][i]
            search_bbox = sample['search_bboxes'][i]
            img = img_tensor.permute(1, 2, 0).numpy()
            x_t, y_t, w_t, h_t = template_bbox.numpy()
            x_s, y_s, w_s, h_s = search_bbox.numpy()
            plt.figure()
            plt.imshow(img)
            # plt.gca().add_patch(plt.Rectangle((x_t, y_t), w_t, h_t, edgecolor='r', facecolor='none', linewidth=2, label='Template bbox'))
            plt.gca().add_patch(plt.Rectangle((x_s, y_s), w_s, h_s, edgecolor='g', facecolor='none', linewidth=2, label='Search bbox'))
            plt.title(f"Search {i} | Seq: {sample['seq_name']} Frame: {sample['frame_ids'][i]}")
            plt.axis('off')

        plt.show()
