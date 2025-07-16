# Placeholder - content will be inserted later
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize
from utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class DiMPTracker:
    def __init__(self, model, image_sz=256, search_factor=2.0, device='cuda'):
        self.model = model.eval().to(device)
        self.device = device
        self.image_sz = image_sz
        self.search_factor = search_factor
        self.transform = Resize((image_sz, image_sz))
        self.to_tensor = ToTensor()

    def _crop_patch(self, image, center, size):
        h, w = image.shape[:2]
        size = int(round(size))
        cx, cy = center
        x1 = int(round(cx - size / 2))
        y1 = int(round(cy - size / 2))
        x2 = x1 + size
        y2 = y1 + size

        # Padding if necessary
        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(0, x2 - w)
        y2_pad = max(0, y2 - h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        patch = image[y1:y2, x1:x2]
        patch = cv2.copyMakeBorder(patch, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_tensor = self.to_tensor(patch).unsqueeze(0).to(self.device)  # [1,3,H,W]
        return self.transform(patch_tensor)

    def initialize(self, frame, init_bbox):
        x, y, w, h = init_bbox
        cx, cy = x + w / 2, y + h / 2
        crop_size = np.sqrt(w * h) * self.search_factor

        template = self._crop_patch(frame, (cx, cy), crop_size)

        self.center = torch.tensor([cx, cy], device=self.device)
        self.size_in_img = crop_size
        self.init_box = torch.tensor([cx, cy, w, h], device=self.device)

        self.model_filter = self.model.initialize(template)

    def track(self, frame):
        # Prepare cropped search image
        search = self._crop_patch(frame, self.center.tolist(), self.size_in_img)

        # Refine model filter
        self.model_filter = self.model.refine(self.model_filter, search)

        # Run classifier (score map) – not used for bbox directly in DiMP, but available
        # Run IoUNet-based prediction
        cx, cy, w, h = self.init_box
        init_box_xyxy = box_cxcywh_to_xyxy(self.init_box.unsqueeze(0))  # [1, 4]
        iou_input = torch.cat([torch.zeros((1, 1), device=self.device), init_box_xyxy], dim=1)  # [1, 5]

        pred_iou = self.model.predict_iou(search, iou_input)  # [1, 1] — not used directly here, optional

        # Assume center stays fixed, return last refined box
        self.center = self.init_box[:2]
        self.target_sz = self.init_box[2:]

        # Scale to original image coordinates
        scale = self.size_in_img / self.image_sz
        cx, cy, w, h = self.init_box.cpu().numpy()
        cx_img = cx * scale + (self.center[0] - self.size_in_img / 2).cpu().numpy()
        cy_img = cy * scale + (self.center[1] - self.size_in_img / 2).cpu().numpy()
        w_img = w * scale
        h_img = h * scale

        return [float(cx_img - w_img / 2), float(cy_img - h_img / 2), float(w_img), float(h_img)]

    def visualize(self, frame, bbox, frame_id=None, wait=1, window_name="DiMP Tracker"):
        x, y, w, h = map(int, bbox)
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if frame_id is not None:
            cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, vis_frame)
        key = cv2.waitKey(wait)
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Tracking interrupted by user.")
