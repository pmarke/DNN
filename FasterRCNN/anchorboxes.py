import numpy as np
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

# Each anchor box will have y1, x1, y2, x2
anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

# The input image is 800 x 800. After the Vgg16 network, the feature size is 50 x 50. Thus, 
# the top right pixel corresponds to 16x16 box in the original image. Thus the center is the 
# pixel at (8,8)
sub_sample = 16
ctr_y = sub_sample / 2.
ctr_x = sub_sample / 2.


for i in range(len(ratios)):
  for j in range(len(anchor_scales)):
    h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
    w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])

    index = i * len(anchor_scales) + j

    anchor_base[index, 0] = ctr_y - h / 2.
    anchor_base[index, 1] = ctr_x - w / 2.
    anchor_base[index, 2] = ctr_y + h / 2.
    anchor_base[index, 3] = ctr_x + w / 2.


fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)

index = 0
ctr =[[0 for x in range(2)] for y in range(len(ctr_x)*len(ctr_y))]
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index][1] = ctr_x[x] - 8
        ctr[index][0] = ctr_y[y] - 8
        index +=1


anchors = np.zeros(((fe_size * fe_size * 9), 4))
index = 0
for c in ctr:
  ctr_y, ctr_x = c
  for i in range(len(ratios)):
    for j in range(len(anchor_scales)):
      h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
      w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
      anchors[index, 0] = ctr_y - h / 2.
      anchors[index, 1] = ctr_x - w / 2.
      anchors[index, 2] = ctr_y + h / 2.
      anchors[index, 3] = ctr_x + w / 2.
      index += 1

# Get the indecies of all of the anchor boxes that are contained inside the image.
index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
    )[0]


# create an array of anchor labels for all of the valid anchors and set them to -1
label = np.empty((len(index_inside), ), dtype=np.int32)
label.fill(-1)

# Create and array of valid anchor boxes
valid_anchor_boxes = anchors[index_inside]
# print(valid_anchor_boxes.shape)

# Create ground truth boxes and object classification labels
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
labels = np.asarray([6, 8], dtype=np.int8) # 0 represents background

# Calculate the intersection over union of the valid anchor boxes with the true boxes.
ious = np.empty((len(valid_anchor_boxes), 2), dtype=np.float32)
ious.fill(0)

for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i  
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2- yb1) * (xb2 - xb1)
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * \
(inter_x2 - inter_x1)
            iou = iter_area / \
(anchor_area+ box_area - iter_area)            
        else:
            iou = 0.
        ious[num1, num2] = iou


gt_argmax_ious = ious.argmax(axis=0)
# print(gt_argmax_ious)
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
# print(gt_max_ious)

argmax_ious = ious.argmax(axis=1)
# print(argmax_ious.shape)
# print(argmax_ious)
max_ious = ious[np.arange(len(index_inside)), argmax_ious]
# print(max_ious)

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
# print(gt_argmax_ious)

pos_iou_threshold  = 0.7
neg_iou_threshold = 0.3



label[max_ious < neg_iou_threshold] = 0 # Assign a zero to all lables that have a max ious less than
label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1

# Up until now I have assigned a 1 to any anchor box that either has the highest Intersection
# over union with any ground truth or has an HIU > 0.7 with any ground truth box.
# I have also assigned a zero to any anchor that intesects a ground truth box and whose highest
# IOU with any ground truth is less than 0.3. All other anchors get a label of -1. 


# We now randomly sample positive and negative samples. 

pos_ratio = 0.5
n_sample = 256

n_pos = pos_ratio * n_sample

# Positive samples
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
    label[disable_index] = -1

# negative samples
n_neg = n_sample - np.sum(label == 1)
# print(n_neg)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
    label[disable_index] = -1



max_iou_bbox = bbox[argmax_ious]
# print(max_iou_bbox.shape)

height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(anchor_locs)




anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label

anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs