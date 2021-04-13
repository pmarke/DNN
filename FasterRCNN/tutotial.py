# https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439

import torch
import torchvision


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  

dummy_img = torch.zeros((1, 3, 800, 800)).float()
dummy_img: torch.Tensor =dummy_img.to(device)
# print(dummy_img)
# print(dummy_img.get_device())

# We want to use vgg16 as the base. 
model = torchvision.models.vgg16(pretrained=True)
model.to(device)
fe = list(model.features)
print(fe)

req_features = []
out_map = dummy_img.clone()
for i in fe:
    out_map = i(out_map)
    if out_map.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = out_map.size()[1]
print(len(req_features)) #30
print(out_channels) # 512


# Convert the list into a Sequential module
faster_rcnn_fe_extractor = torch.nn.Sequential(*req_features)


mid_channels = 512
in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
n_anchor = 9 # Number of anchors at each location
conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
conv1.to(device)
reg_layer = torch.nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0) # The 4 corresponds to the encoding of the box's coordinates
reg_layer.to(device)
cls_layer = torch.nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0) # The 2 represents the probability of object or not object for each anchor
cls_layer.to(device)

# conv sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()
# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()
# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()


x = conv1(out_map) # out_map is obtained in section 1
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

# This is of size 1x22500x4
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

# This is of size 1x50x50x18
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()

# This is of size 1x22500. 
objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1) 

# This is of size 1x22500x2
pred_cls_scores  = pred_cls_scores.view(1, -1, 2)

# Non maximum supression (NMS)

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# non maximum supression
# - Take all the roi boxes [roi_array]
# - Find the areas of all the boxes [roi_area]
# - Take the indexes of order the probability score in descending order [order_array]
# keep = []
# while order_array.size > 0:
#   - take the first element in order_array and append that to keep  
#   - Find the area with all other boxes
#   - Find the index of all the boxes which have high overlap with this box
#   - Remove them from order array
#   - Iterate this till we get the order_size to zero (while loop)
# - Ouput the keep variable which tells what indexes to consider.







