import torch 
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes 
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO 
import torchvision.transforms.v2 as v2 
from torchvision import tv_tensors
from model import YOLOv11
import os
import torch.nn.functional as F 
import torchvision

from torchvision.ops import box_convert, box_iou

IMG_PATH_VAL = "/home/artemis/DNN/datasets/coco/val2017"
IMG_PATH_TRAIN = "/home/artemis/DNN/datasets/coco/train2017"
ANNOTATION_PATH_VAL ="/home/artemis/DNN/datasets/coco/annotations_trainval2017/annotations/instances_val2017.json"
ANNOTATION_PATH_TRAIN ="/home/artemis/DNN/datasets/coco/annotations_trainval2017/annotations/instances_train2017.json"

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.transforms = transforms
        dataset = datasets.CocoDetection(img_dir, annotation_file)
        dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels"))

        # Load COCO API to get label mapping
        self.coco = COCO(annotation_file)
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}

        self.cat_id_to_contiguous_id = {
            cat_id: idx for idx, cat_id in enumerate(self.coco.getCatIds())
        }
        self.contiguous_id_to_cat_id = {v: k for k, v in self.cat_id_to_contiguous_id.items()}

        # Filter out samples without "boxes" or "labels"
        self.dataset = [
            (img, target) for img, target in dataset
            if "labels" in target and "boxes" in target and len(target["boxes"]) > 0
        ]

        if len(self.dataset) == 0:
            raise ValueError("No valid samples found in the dataset.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #The target is a dictionary containing the keys 'boxes' and 'lables'. 
        #the boxes are a type BoundingBox with format XYXY (tl and br)"
        img, target = self.dataset[idx]

        if self.transforms:
            img, target = self.transforms(img, target)

        # Safety check: wrap_dataset_for_transforms_v2 should give dicts with 'boxes' and 'labels'
        if "labels" not in target or "boxes" not in target:
            raise ValueError(f"Target format invalid at index {idx}. Got: {target}")

        # Map COCO category_id to contiguous [0..79]
        target["labels"] = torch.tensor([
            self.cat_id_to_contiguous_id[c.item()] for c in target["labels"]
        ], dtype=torch.long) 

        return img, target

    
    def get_label_names(self, labels_tensor):
        return [self.cat_id_to_name[label.item()] for label in labels_tensor]

    def display_sample(self, img, target):

        if(not isinstance(img, torch.Tensor)):
            # Convert PIL Image to Tensor
            img = ToTensor()(img)


        img_tensor = (img * 255).to(torch.uint8) 

        label_strings = self.get_label_names(target['labels'])
        img_with_boxes = draw_bounding_boxes(img_tensor, boxes=target['boxes'], labels=label_strings, width=2, colors="red")

        plt.imshow(to_pil_image(img_with_boxes))
        plt.axis("off")
        plt.show()


def collate_fn(batch):
    """
    Custom collate function for object detection.

    Stacks images and returns targets as a list of dictionaries.
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    return images, list(targets)

def CreateCocoDataLoader(img_dir, annotation_file, batch_size=32, shuffle=True):


    transforms = v2.Compose([v2.Resize((640, 640)),
                             v2.ToImage(), 
                             v2.ToDtype(torch.float32, scale=True), 
                             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = CocoDataset(img_dir, annotation_file, transforms=transforms)

    # img, target = dataset[0]
    # dataset.display_sample(img, target)

    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle, collate_fn=collate_fn) 


def dfl_loss(pred_logits, target, n_bins=16):
    """
    Compute Distribution Focal Loss (DFL) for one side of the box.
    
    Args:
        pred_logits: Tensor of shape [N, n_bins] — raw logits for one box side (e.g., left).
        target: Tensor of shape [N] — real-valued ground-truth distances (in bin units).
        n_bins: Number of discrete bins (default: 16 or 32).

    Returns:
        Scalar loss value (averaged over batch).
    """
    # Clamp target to ensure it's within [0, n_bins)
    target = target.clamp(0, n_bins - 1 - 1e-6)

    # Get left and right bin indices
    l = target.floor().long()  # lower bin index
    r = l + 1                  # upper bin index

    # Linear interpolation weights
    wl = r.float() - target    # weight for left bin
    wr = target - l.float()    # weight for right bin

    # Get log probabilities
    log_probs = F.log_softmax(pred_logits, dim=1)  # [N, n_bins]

    # Negative log-likelihood loss: only on two closest bins
    loss = - (log_probs[range(len(target)), l] * wl +
              log_probs[range(len(target)), r.clamp(max=n_bins - 1)] * wr)

    return loss.mean()


class Integral(nn.Module):
    def __init__(self, n_bins=16):
        super().__init__()
        # Create a tensor [0, 1, ..., n_bins - 1] to project the softmax output into a scalar
        self.register_buffer('project', torch.linspace(0, n_bins - 1, n_bins))

    def forward(self, x):
        # x shape: [..., 4, n_bins], for 4 sides of the box: [left, top, right, bottom]
        x = F.softmax(x, dim=-1)  # Convert to a probability distribution over bins
        return (x * self.project).sum(dim=-1)  # Weighted sum gives expected distance for each side


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        """
        alpha: balancing factor for positive/negative examples
        gamma: focusing parameter to down-weight easy examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, C] (raw class predictions for N samples)
        # targets: [N] (true class indices in 0 to C-1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # base cross-entropy
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss  # focal modulation

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal  # no reduction


class DetectionLoss(nn.Module):
    def __init__(self, n_classes=80, n_bins=16, box_weight=1.0, cls_weight=1.0):
        """
        n_classes: number of object classes (e.g., 80 for COCO)
        n_bins: number of bins used in Distribution Focal Loss
        box_weight / cls_weight: weighting for combining losses


        The current DetectionLoss implementation doesn't explicitly 
        supervise background (negative) locations because it assumes 
        that only positive grid cells (those assigned to targets) are 
        responsible for learning — which is standard in anchor-free object 
        detectors like YOLOv5–v8, FCOS, and GFL.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.integral = Integral(n_bins)
        self.focal_loss = FocalLoss()
        self.box_weight = box_weight
        self.cls_weight = cls_weight

    def forward(self, outputs, targets, anchors=None):
        """
        outputs: list of 3 three tensors (b, n_classes + 4*n_bins,lw,lh) for each FPN level
        targets: list of ground truth boxes per image, each is a tensor [N_i, 6]
            - columns: [class, x_center, y_center, width, height, layer_index]
        anchors: list of anchor center coordinates (not used here directly, but useful for GT assignment)

        returns:
            total_loss, total_cls_loss, total_box_loss
        """
        total_cls_loss, total_box_loss = 0.0, 0.0

        # iterate through each layer
        for layer_idx, feat in enumerate(outputs):
            B, C, H, W = feat.shape
            dist_pred = feat[:, :4 * self.n_bins, :, :]  # DFL logits
            cls_pred = feat[:, 4 * self.n_bins:, :, :]   # class logits


            # Rearrange prediction tensors to [B, H, W, C] and [B, H, W, 4, n_bins]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()          # [B, H, W, C]
            dist_pred = dist_pred.permute(0, 2, 3, 1).contiguous()        # [B, H, W, 4 * n_bins]
            dist_pred = dist_pred.view(B, H, W, 4, self.n_bins)           # [B, H, W, 4, n_bins]
            device = dist_pred.device

            #iterate through each batch
            for b in range(B):
                image_targets = targets[b]  # shape [N, 6]
                layer_targets = image_targets[image_targets[:, -1] == layer_idx]

                if layer_targets.numel() == 0:
                    continue  # No targets for this level/layer

                for t in layer_targets:
                    # Parse ground truth values
                    cls, cx, cy, w, h, _ = t
                    cls = int(cls)
                    if not (0 <= cls < self.n_classes):
                        raise ValueError(f"Invalid class index: {cls}")

                    if(cx < 0 or cx > 1):
                        print('large cx: ', cx) 
                    if(cy < 0 or cx > 1): 
                        print('large cy: ', cy)

                    # Scale center x/y to the feature map resolution
                    fx = min(max(int(cx * W), 0), W - 1)
                    fy = min(max(int(cy * H), 0), H - 1)

                    # Select predictions at (fy, fx) location
                    pred_logits = cls_pred[b, fy, fx]     # [C] classification logits
                    pred_distr = dist_pred[b, fy, fx]     # [4, n_bins] DFL logits
                    # pred_box = self.integral(pred_distr)  # [4] decoded distances

                    # Compute ground truth box in l/t/r/b format relative to center
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    l = max(cx - x1, 0)
                    t_ = max(cy - y1, 0)
                    r = max(x2 - cx, 0)
                    b_ = max(y2 - cy, 0)
                    gt_box = torch.tensor([l * W, t_ * H, r * W, b_ * H], device=device)

                    # Compute classification loss for this point
                    total_cls_loss += self.focal_loss(
                        pred_logits[None],  # shape [1, C]
                        torch.tensor([cls], dtype=torch.long, device=pred_logits.device)
                    )

                    # Compute bounding box regression loss (L1 between predicted and GT distances)
                    # total_box_loss += F.l1_loss(pred_box, gt_box, reduction='mean')

                for i in range(4):
                    pred_logits = pred_distr[i].unsqueeze(0)  # [1, n_bins]
                    target = gt_box[i].unsqueeze(0)           # [1]
                    dfl = dfl_loss(pred_logits, target, n_bins=self.n_bins)
                    total_box_loss += dfl

        # Combine losses with weights
        total_loss = self.cls_weight * total_cls_loss + self.box_weight * total_box_loss
        return total_loss, total_cls_loss, total_box_loss


def assign_targets_to_levels_yolov8(targets, image_size, grid_sizes, strides, center_radius=2.5, scale_ranges=None):
    """
    Modified YOLOv8-style assignment function.
    Each GT box is assigned to grid cells within a radius around its center
    in the appropriate FPN level (P3–P5).

    Args:
        targets: list of tensors [N, 5] = [cls, cx, cy, w, h] in normalized coords
        image_size: int, input size (e.g., 640)
        grid_sizes: list of ints, H/W for each level (e.g., [80, 40, 20])
        strides: list of ints, stride per level (e.g., [8, 16, 32])
        center_radius: float, radius (in pixels) for center sampling
        scale_ranges: optional list of (min, max) ranges per level (in pixels)

    Returns:
        list of tensors per image, each [N, 6] = [cls, cx, cy, w, h, level_idx]
    """
    if scale_ranges is None:
        # Default scale ranges in pixels for P3, P4, P5
        scale_ranges = [(0, 64), (64, 128), (128, 1e5)]

    assigned = []

    for img_targets in targets:
        if len(img_targets) == 0:
            assigned.append(torch.zeros((0, 6)))
            continue

        out = []

        for level_idx, (stride, grid_size, (min_s, max_s)) in enumerate(zip(strides, grid_sizes, scale_ranges)):
            # Calculate box size in pixels
            boxes = img_targets[:, 1:5]  # [cx, cy, w, h]
            wh_pixels = boxes[:, 2:4] * image_size
            box_scales = torch.sqrt(wh_pixels[:, 0] * wh_pixels[:, 1])

            # Filter boxes by scale range
            mask = (box_scales >= min_s) & (box_scales < max_s)
            if not mask.any():
                continue

            # select the targets for the specific feature level
            level_targets = img_targets[mask]
            cx, cy, w, h = level_targets[:, 1], level_targets[:, 2], level_targets[:, 3], level_targets[:, 4]

            # Compute center in pixels
            cx_pix = cx * image_size
            cy_pix = cy * image_size

            # Convert center radius to feature map grid units
            radius_feat = center_radius / stride

            for i in range(len(level_targets)):
                gx = cx_pix[i] / stride
                gy = cy_pix[i] / stride

                # Compute surrounding grid cell indices
                x0 = int(torch.floor(gx - radius_feat))
                x1 = int(torch.ceil(gx + radius_feat))
                y0 = int(torch.floor(gy - radius_feat))
                y1 = int(torch.ceil(gy + radius_feat))

                # Clamp to grid
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, grid_size - 1)
                y1 = min(y1, grid_size - 1) 

                # Assign this GT to multiple grid cells around its center
                for gy_idx in range(y0, y1 + 1):
                    for gx_idx in range(x0, x1 + 1):
                        # Compute normalized coordinates of grid cell center
                        norm_cx = (gx_idx + 0.5) * stride / image_size
                        norm_cy = (gy_idx + 0.5) * stride / image_size
                        row = torch.tensor([
                            level_targets[i, 0],  # class
                            norm_cx,
                            norm_cy,
                            level_targets[i, 3],  # width
                            level_targets[i, 4],  # height
                            float(level_idx)      # level index
                        ])
                        out.append(row)

        if len(out) > 0:
            assigned.append(torch.stack(out, dim=0))
        else:
            assigned.append(torch.zeros((0, 6)))

    return assigned

def convert_targets_for_yolo(batched_targets, image_size):
    """
    Converts a batch of COCO-style targets to YOLO format.
    
    Args:
        batched_targets: list of target dicts, each with keys 'boxes', 'labels'
        image_size: int, assumed square input (e.g., 640)
    
    Returns:
        List of tensors, one per image: [N, 5] = [class, cx, cy, w, h]
    """
    yolo_targets = []
    for t in batched_targets:
        boxes = t["boxes"]  # [N, 4] xyxy
        labels = t["labels"]


        # Convert boxes from xyxy to cxcywh and normalize
        if boxes.numel() > 0:
            cxcywh = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
            cxcywh /= image_size
        else:
            cxcywh = torch.zeros((0, 4), device=boxes.device) 
            print('no boxes found')

        # Create the YOLO target tensor: [class, cx, cy, w, h]
        if labels.numel() > 0:
            yolo_target = torch.cat((labels.unsqueeze(1).float(), cxcywh), dim=1)
            yolo_targets.append(yolo_target)
        else:
            # Handle images with no objects
            yolo_targets.append(torch.zeros((0, 5), device=boxes.device)) 
            print("no labels found")

 
    return yolo_targets





def trainModel(train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               model: YOLOv11,
               num_train_batches: int):

    print("training model")
    model.train()

    # Set loss function
    criterion = DetectionLoss(n_classes=80, n_bins=16)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()  # Corrected GradScaler initialization

    image_size = 640
    strides = [8, 16, 32]
    grid_sizes = [image_size // s for s in strides]

    print("starting training with num_epochs:", num_epochs)

    for epoch in range(num_epochs):
        for i, (images, coco_targets) in enumerate(train_loader):
            images = images.to(device) 

            # Convert COCO-style targets to [cls, cx, cy, w, h] (normalized)
            yolo_targets = convert_targets_for_yolo(coco_targets, image_size)

            # Assign targets to levels/grid cells
            # list of tensors per image, each [N, 6] = [cls, cx, cy, w, h, level_idx]
            assigned_targets = assign_targets_to_levels_yolov8(
                yolo_targets, image_size=image_size, grid_sizes=grid_sizes, strides=strides)

            # Move each image's assigned targets to the correct device
            assigned_targets = [t.to(device) for t in assigned_targets]
            print(assigned_targets)

            optimizer.zero_grad()  # Ensure gradients are cleared before forward pass

            with torch.cuda.amp.autocast(True):  # Use mixed precision for faster training
                # outputs a list of tensors for each layer of shape (b,n_class + 4*n_bins, lw,lh)
                outputs = model(images)
                loss, cls_loss, box_loss = criterion(outputs, assigned_targets)

            # print(f"Assigned targets: {assigned_targets}")
            # print(f"Predictions: {outputs}")

            scaler.scale(loss).backward()  # Scale loss for mixed precision
            scaler.step(optimizer)
            scaler.update()

            print(f'Batch [{i+1}/{num_train_batches}], Loss: {loss.item():.4f}, '
                  f'Cls: {cls_loss.item():.4f}, Box: {box_loss.item():.4f}')

            torch.cuda.empty_cache()  # Clear cache to avoid memory issues
            del images, outputs

        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}] completed.\n')




def decode_predictions(outputs, strides, n_bins=16, score_threshold=0.3):
    """
    Decodes YOLOv8-style model outputs to bounding boxes.

    Args:
        outputs: List of (cls_logits, dist_preds) per FPN level.
        strides: List of strides for each level.
        n_bins: Number of bins in DFL.
        score_threshold: Minimum confidence to keep predictions.

    Returns:
        List of detections: [B, N, 6] = [x1, y1, x2, y2, score, class]
    """
    results = []
    integral = Integral(n_bins=n_bins).to(outputs[0][0].device)

    for b in range(outputs[0][0].shape[0]):
        all_boxes = []
        all_scores = []
        all_labels = []

        for level, (cls_logits, dist_pred) in enumerate(outputs):
            B, C, H, W = cls_logits.shape
            stride = strides[level]

            cls_logits = cls_logits[b].permute(1, 2, 0).reshape(-1, C)        # [H*W, C]
            dist_pred = dist_pred[b].permute(1, 2, 0).reshape(H*W, 4, n_bins) # [H*W, 4, n_bins]

            scores = torch.softmax(cls_logits, dim=-1)                        # class probabilities
            confidences, labels = scores.max(dim=1)                          # best class per location

            keep = confidences > score_threshold
            if keep.sum() == 0:
                continue

            scores = confidences[keep]
            labels = labels[keep]
            preds = dist_pred[keep]

            box_dists = integral(preds)  # [N, 4]: [l, t, r, b]
            grid_y, grid_x = torch.div(torch.arange(H*W)[keep], W, rounding_mode='floor'), torch.arange(H*W)[keep] % W

            x_center = (grid_x + 0.5) * stride
            y_center = (grid_y + 0.5) * stride

            x1 = x_center - box_dists[:, 0]
            y1 = y_center - box_dists[:, 1]
            x2 = x_center + box_dists[:, 2]
            y2 = y_center + box_dists[:, 3]

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)
            result = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
        else:
            result = torch.zeros((0, 6), device=cls_logits.device)

        results.append(result)

    return results

def validate_model(model, val_loader, image_size=640, n_bins=16, iou_threshold=0.5):
    """
    Evaluates model on the validation set using IoU thresholding.

    Args:
        model: YOLOv8-style model
        val_loader: DataLoader returning (image, target) with COCO format
        image_size: int, assumed square
        n_bins: number of DFL bins
        iou_threshold: float, IoU threshold for true positive

    Prints:
        Precision, Recall, and number of matched GTs
    """
    model.eval()
    strides = [8, 16, 32]
    total_preds = 0
    total_gts = 0
    true_positives = 0

    with torch.no_grad():
        for images, coco_targets in val_loader:
            images = images.to(device)
            yolo_targets = convert_targets_for_yolo(coco_targets, image_size)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)

            decoded = decode_predictions(outputs, strides=strides, n_bins=n_bins)

            for preds, gt in zip(decoded, yolo_targets):
                if len(preds) == 0 or len(gt) == 0:
                    continue

                pred_boxes = preds[:, :4]  # [N, 4]
                gt_boxes = gt[:, 1:5] * image_size  # un-normalize
                ious = box_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]

                # Greedy matching
                matched_gt = set()
                for i in range(ious.size(0)):
                    max_iou, idx = ious[i].max(0)
                    if max_iou > iou_threshold and idx.item() not in matched_gt:
                        true_positives += 1
                        matched_gt.add(idx.item())

                total_preds += len(preds)
                total_gts += len(gt)

    precision = true_positives / total_preds if total_preds > 0 else 0
    recall = true_positives / total_gts if total_gts > 0 else 0
    print(f"Validation Results:")
    print(f"  True Positives: {true_positives}")
    print(f"  Total Predictions: {total_preds}")
    print(f"  Total Ground Truths: {total_gts}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")


if __name__ =='__main__':

    modelPath = "/home/artemis/DNN/yolo11/model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())

    batch_size = 4
    coco_dataloader_val = CreateCocoDataLoader(IMG_PATH_VAL, ANNOTATION_PATH_VAL, batch_size=batch_size)

    # coco_dataloader_train = CreateCocoDataLoader(IMG_PATH_TRAIN, ANNOTATION_PATH_TRAIN, batch_size=batch_size)


    # num_epochs = 20
    num_epochs = 1
    learning_rate = 0.01

    num_val_batches = len(coco_dataloader_val)
    print("num_val_batches", coco_dataloader_val)

    # The resnet model is broken up into 4 main blocks where each block decreases the size by a factor of two 
    # and increases the number of channels by a factor of two.
    model = YOLOv11().build_model(version='n', num_classes=80).to(device)

    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))


    trainModel(coco_dataloader_val, coco_dataloader_val, model, num_val_batches)

    torch.save(model.state_dict(), modelPath)

    # testModel(test_loader,model)

    # Tensorboard embedding projector: https://www.youtube.com/watch?v=RLqsxWaQdHE&ab_channel=AladdinPersson
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html



