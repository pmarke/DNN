# Placeholder - content will be inserted later
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .iou_net import IoUNet
from .residual_function import ResidualFunction
from utils.heatmap import gaussian_heatmap
import torch.nn.init as init

class DiMPNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super().__init__()

        from torchvision.models import ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT if backbone_pretrained else None)
        # Use up to layer3 (output: [B, 256, 14, 14] for 224x224 input)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        self.cls_feat = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )

        self.init_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.update_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        self.score_head = nn.Conv2d(64, 1, kernel_size=1)
        self.iou_predictor = IoUNet(in_channels=64)

        self.residual_fn = ResidualFunction(spatial_size=17, rbf_centers=50)

        # Initialize weights using Kaiming method
        for m in self.cls_feat.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.init_conv.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.update_conv.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.score_head.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for m in self.residual_fn.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
      

    def forward(self, template, search, iters=5, return_bbox=False, candidate_boxes=None):
        # print("template min/max/mean:", template.min(), template.max(), template.mean())
        z = self.backbone(template)
        x = self.backbone(search)

        # print("z",z[0][0])
        # print("z",z[0][5])
        # print("x",x[0][0])
        # print("x",x[0][5])

        zc = self.cls_feat(z)
        xc = self.cls_feat(x)
        # print("zc",zc[0][0])
        # print("xc",xc[0][0])
        # print("zc",zc.shape)
        # print("xc",xc.shape)
        model = self.init_conv(zc)

        # this update is hurting it so I am not using it
        # for _ in range(iters):
        #     joint = torch.cat([model, model], dim=1)
        #     delta = self.update_conv(joint)
        #     model = model - delta 

        # print("model", model.shape)
        # print("model",model[0][0])

        # temp = F.conv2d(x,z,padding=0)

        # max_idx = torch.argmax(temp)
        # max_idx_unraveled = torch.unravel_index(max_idx, temp.shape[2:])
        # max_height_idx, max_width_idx = max_idx_unraveled
        # print("Height index of max value in temp:", max_height_idx)
        # print("Width index of max value in temp:", max_width_idx)
        # print("Index of max value in temp:", max_idx)
        # print("temp", temp)
        

        # Apply filter to search features (cross-correlation)
        # You may need to upsample model to match xc spatial size, or use a conv2d
        # Here is a simple depthwise correlation:
        # For each batch, apply model[b] as a filter to xc[b]
        score = []
        for b in range(xc.shape[0]):
            # [1, 64, H, W], [64, 1, kH, kW]
            score_b = F.conv2d(xc[b:b+1], model[b].unsqueeze(1), padding=0, groups=64)
            score.append(score_b)
        score = torch.cat(score, dim=0)  # [B, 64, outH, outW]
        # print("score",score[0][0])
        # Apply 2D softmax on the score
        score = self.score_head(score)   # [B, 1, outH, outW]
        # print("score a",score[0][0])

        score = torch.sigmoid(score)
        # score = F.softmax(score.view(score.shape[0], score.shape[1], -1), dim=-1).view_as(score)

        # print("score b", score.shape)
        # print(score[0])


        if not return_bbox:
            return score
        if candidate_boxes is None:
            raise ValueError("candidate_boxes must be provided when return_bbox=True")
        iou_preds = self.iou_predictor(xc, candidate_boxes)
        return score, iou_preds

    def initialize(self, template):
        with torch.no_grad():
            z_feat = self.backbone(template)
            z_cls = self.cls_feat(z_feat)
            return self.init_conv(z_cls)

    def refine(self, model_filter, search):
        x_feat = self.backbone(search)
        x_cls = self.cls_feat(x_feat)
        joint = torch.cat([x_cls, model_filter], dim=1)
        delta = self.update_conv(joint)
        return model_filter - delta

    def predict_iou(self, search, boxes):
        with torch.no_grad():
            feat = self.backbone(search)
            x_cls = self.cls_feat(feat)
            return self.iou_predictor(x_cls, boxes)
        
    # def compute_residual_loss(self, score_map, centers):
    #     res = self.residual_fn(score_map, centers)  # [B, H, W]
    #     return (res ** 2).mean()
    
    # def compute_residual_loss(self, score_map, bbox, search_img_size):
    #     # score_map: [B, 1, H, W]
    #     # bbox: [B, 4] in search image coordinates
    #     # search_img_size: int (e.g., 256)

    #     B, _, H, W = score_map.shape
    #     centers = []

    #     for i in range(B):
    #         # bbox[i]: [x, y, w, h] in search image coordinates
    #         cx_img = bbox[i][0] + bbox[i][2] / 2
    #         cy_img = bbox[i][1] + bbox[i][3] / 2
    #         # Map center from search image to score map coordinates
    #         cx_map = cx_img / search_img_size * W
    #         cy_map = cy_img / search_img_size * H
    #         centers.append([cx_map, cy_map])

    #     centers = torch.tensor(centers, dtype=score_map.dtype, device=score_map.device)  # [B, 2]
    #     # Pass centers to residual_fn
    #     res = self.residual_fn(score_map.squeeze(1), centers)  # [B, H, W]
    #     return (res ** 2).mean()

    def compute_residual_loss(self, score_map, bbox, search_img_size, sigma=3):
        # score_map: [B, 1, H, W]
        # bbox: [B, 4] in search image coordinates
        # search_img_size: int (e.g., 256)
        B, _, H, W = score_map.shape
        heatmaps = []
        centers = []
        for i in range(B):
            # bbox[i]: [x, y, w, h] in search image coordinates
            cx_img = bbox[i][0] + bbox[i][2] / 2
            cy_img = bbox[i][1] + bbox[i][3] / 2
            # Map center from search image to score map coordinates
            cx_map = cx_img / search_img_size * W
            cy_map = cy_img / search_img_size * H
            centers.append([cx_map, cy_map])
            heatmap = gaussian_heatmap(W, (cx_map, cy_map), sigma, device=score_map.device)  # shape [H, W]
            heatmaps.append(torch.tensor(heatmap, dtype=score_map.dtype, device=score_map.device))
        heatmaps = torch.stack(heatmaps, dim=0).unsqueeze(1)  # [B, 1, H, W]
        # Compute MSE loss
        print("score_map", score_map[0])
        print("heatmaps", heatmaps[0])
        print('boxes', bbox[0])
        print('centers', centers[0])
        return torch.nn.functional.mse_loss(score_map, heatmaps)