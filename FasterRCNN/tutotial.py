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
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]
print(len(req_features)) #30
print(out_channels) # 512


# Convert the list into a Sequential module
faster_rcnn_fe_extractor = torch.nn.Sequential(*req_features)


mid_channels = 512
in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
n_anchor = 9 # Number of anchors at each location
conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
conv1.to(device)
reg_layer = torch.nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0)
reg_layer.to(device)
cls_layer = torch.nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0) ## I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
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

pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
pred_cls_scores  = pred_cls_scores.view(1, -1, 2)




