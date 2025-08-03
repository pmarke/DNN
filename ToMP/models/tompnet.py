import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.nn.init as init
from utils.heatmap import generate_heatmap


class ConvBlock(nn.Module) :

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, group =1, activation = nn.SiLU()):
        super().__init__()

        # With convolutional layers, Batch normalization is applied to each channel. This means that there are separate (mean, std) parameters for each 
        # channel. The idea is that the same feature map is convolved with the input to produce a single channel, so the outputs in a channel should have 
        # similar information and thus similar statistics.
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=group, bias = False),
                        nn.BatchNorm2d(out_channels),
                        activation)

    def forward(self, x):
        return self.conv(x)

class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = ConvBlock(ch, ch + self.dim_key * num_head * 2, kernel_size=1, stride=1, padding=0, activation= torch.nn.Identity())

        self.conv1 = ConvBlock(ch, ch, activation = torch.nn.Identity(), kernel_size=3, padding=1, group=ch)
        self.conv2 = ConvBlock(ch, ch, activation = torch.nn.Identity(), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
    

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)

class MultiheadAttentionFFN(nn.Module):

    def __init__(self, ch, num_head):
        '''
        Multi-head attention with FlashAttention using convolutional QKV projection
        '''
        super().__init__()

        # self.mha = ConvMultiHeadFlashAttention(ch,num_head)
        self.mha = Attention(ch,num_head)
        self.ffn = torch.nn.Sequential(ConvBlock(ch, ch * 2, activation=torch.nn.SiLU(), kernel_size=1, stride=1, padding=0),
                                                ConvBlock(ch * 2, ch, activation=torch.nn.Identity(),kernel_size=1, stride=1, padding=0))

    def forward(self,x): 

        x = x + self.mha(x) 
        return x + self.ffn(x)

class ToMPNet(nn.Module):
    def __init__(self, backbone_pretrained=True, sigma = 1/4):
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

        # This is the stride from the window size to the feature size that is the output of the backbone
        self.stride = 16 
        self.sigma = 1.0/4

        # This is applied to a Gaussian map for each training template
        self.foreground_encoding = nn.Conv2d(1,256, kernel_size=1, stride=1, padding=0,bias=False)
        # This is applied to a ones map for each test window
        self.test_encoding = nn.Conv2d(1,256, kernel_size=1, stride=1, padding=0,bias=False)


        # input size is 4x14x14=784
        self.box_encoding = nn.Sequential(nn.Conv2d(4,64,kernel_size=3, padding=1, bias=False),  # 12544 = 256 * 14 * 14
                                          nn.BatchNorm2d(64),
                                          nn.SiLU(),
                                          nn.Conv2d(64,256,kernel_size=3, padding=1, bias=False), # 50175 = 14x14x256
                                          nn.BatchNorm2d(256),
                                          nn.SiLU(), 
                                          nn.Conv2d(256,256,kernel_size=3, padding=1, bias=False)) 
        
        
        self.encoder = MultiheadAttentionFFN(ch=256, num_head=8)  # 256 is the output channel size of the backbone
        self.decoder = MultiheadAttentionFFN(ch=256, num_head=8)  # 256 is the output channel size of the backbone

        # self.weight_layer = nn.Conv2d(256, 2 * 256, kernel_size=3, padding=1, groups=256, bias=True)
        self.weight_layer = nn.Conv2d(256,2*256,kernel_size=3, padding=1, bias=True) 

        self.bbox_conv = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256,4, kernel_size=1, stride=1, padding=0, bias=True),  # Output 4 channels for bounding box regression
            nn.Sigmoid()
        )

        # Initialize weights using Kaiming method
        for m in self.foreground_encoding.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.test_encoding.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.box_encoding.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.bbox_conv.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
      
    def forward(self, training_window_list, test_window, training_bbox_list=None):
            '''
            param: training_window_list: a list of Tensors of shape [B, C, H, W] for training and is the search window 
                    The list is represents a sequence of images. The order of the bactch is preserved across the sequence so that 
                    the first item in the batch corresponds to the first video. 
            param: test_window: Tensor of shape [B, C, H, W] for testing and is the search window
            param: training_bbox_list: list of Tensors of shape [B, 4] containing the bounding boxes of the target in the 
                training search window in the order [x,y, w, h]. The list represents a sequence of images. 
            '''
            [B, C, H, W] = test_window.shape


            x = self.backbone(test_window)
            [B,fc, fw, fh] = x.shape
            tmp = torch.ones((B,1,fw,fh), device=x.device, requires_grad=False)
            test_encoding = self.test_encoding(tmp)  # [B, 256, fw, fh]
            x = x+ test_encoding
            x = x.view(B, fc,-1) # flatten to [B, fc, fw*fh]

            zs = [] 
            for ii in range(len(training_window_list)):
                training_window = training_window_list[ii]
                training_bbox = training_bbox_list[ii]
                z = self.backbone(training_window)
                scaled_bbox = self.scale_bounding_boxes(training_bbox, 1.0/self.stride)
                # Scaled box in the form [cx,cy,w,h]
                heatmaps = generate_heatmap(scaled_bbox, fw,sigma=self.sigma) 
                heatmaps = heatmaps.to(training_window.device)
                foreground_encoding = self.foreground_encoding(heatmaps)  # [B, 256, fw, fh]

                box_encoding = self.generate_target_extent_encoding(training_bbox, self.stride, W, H, fw, fh) 
                box_encoding = self.box_encoding(box_encoding)  # [B, 256, fw, fh]
                # box_encoding = box_encoding.view(B, 256, fw, fh)  # [B, 256, fw, fh]

                z  = z+ foreground_encoding
                z = z+ box_encoding
                zs.append(z.view(B,fc, -1)) # flatten to [B, fc, fw*fh]

            if len(zs) == 1:
                zs.append(zs[0])  # Duplicate the single item to make zs have two items

            
            # Need to concatenate zs along the last dimension
            z_concat = torch.cat(zs, dim=-1)  
            output = torch.cat([z_concat, x], dim=-1).unsqueeze(2)
            # output is now [B, fc,1, fw*fh*3] 

            output = self.encoder(output)  # Apply the encoder
            output = output.squeeze(2)

            z_test = output[:, :, :fw * fh]  # Extract the test encoding part
            z_test = z_test.view(B, fc, fw, fh)  # Reshape to [B, fc, fw, fh]

            # The size of the output is [B, fc, fw*fh*3 + 1] since we added the test_encoding weight 
            test_encoding_weight = self.test_encoding.weight.view(-1,1) 
            test_encoding_weight = test_encoding_weight.expand(B, -1,1)

            output = torch.cat([output, test_encoding_weight], dim=-1).unsqueeze(2) 
            output = self.decoder(output).squeeze(2)  

            # weight size is [B,C,1]
            weight = output[:, :, -1].view(B,fc,1,1) 
            weight = self.weight_layer(weight)  # [B, 2*fc]
            weight_cls = weight[:, :fc]  # [B, fc]
            weight_bbox = weight[:, fc:]  # [B, fc]

            # print("z_test shape", z_test.shape)
            # print("weight_cls shape", weight_cls.shape)
            # print("weight_bbox shape", weight_bbox.shape)

            # Convolve z_test with weight_cls and weight_bbox
            y_cls = torch.einsum('bchw,bc->bhw', z_test, weight_cls.view(B,-1))  # [B, H, W]
            y_bbox = torch.einsum('bchw,bc->bhw', z_test, weight_bbox.view(B,-1))  # [B, H, W]

            bbox = z_test * y_bbox.unsqueeze(1)  # Pointwise multiplication, unsqueeze to match dimensions

        
            bbox = self.bbox_conv(bbox)  # Apply the bounding box convolution and get shape [B,4, fw, fh] 


            # Find the maximum location in y_cls for each batch
            max_indices = torch.argmax(y_cls.view(B, -1), dim=-1)  # Flatten y_cls to [B, H*W] and find max indices
            max_locations = torch.stack([max_indices // fw, max_indices % fw], dim=-1)  # Convert to (h, w) indices

            # Extract the ltrb values from bbox at the maximum locations
            ltrb = []
            for b in range(B):
                h, w = max_locations[b]
                l, t, r, b = bbox[b, :, h, w]
                l = l*fw * self.stride
                r = r*fw * self.stride
                t = t*fh * self.stride
                b = b*fh * self.stride
                ltrb.append(torch.tensor([l, t, r, b], device=bbox.device))
            ltrb = torch.stack(ltrb, dim=0)  # Stack to [B, 4] to the scale of the search window  
            
            max_locations = max_locations * self.stride  # Scale by stride
            max_locations = max_locations.flip(-1)  # Change order to [cx, cy]

            return y_cls, ltrb, max_locations
            

    def scale_bounding_boxes(self, boxes, scale_factor):
            """
            Scale bounding boxes by a factor.
            
            Args:
                boxes: Tensor of shape [B, 4] with bounding boxes in (x, y, w, h) format.
                scale_factor: Scaling factor to apply to the bounding boxes.
            
            Returns:
                Scaled bounding boxes in the same format.
            """
            x, y, w, h = boxes.unbind(-1)
            cx = x + w / 2.0
            cy = y + h / 2.0
            cx *= scale_factor
            cy *= scale_factor
            w *= scale_factor
            h *= scale_factor
            return torch.stack([cx, cy, w, h], dim=-1) 

# def generate_target_extent_encoding(self,boxes, scale_factor, search_window_width, search_window_height, feature_width, feature_height):
#     """
#     Generate target extent encoding for the given bounding boxes.
    
#     Args:
#         boxes: Tensor of shape [B, 4] with bounding boxes in (cx, cy, w, h) format. Bounding boxes are in 
#                 the search window frame
#         scale_factor: Scaling factor such that search_window_width/height = scale_factor * feature_width/height.
#         search_window_width: Width of the search window.
#         search_window_height: Height of the search window.
#         feature_width: Width of the feature map.
#         feature_height: Height of the feature map.
    
    
#     """
    
#     cx, cy, w, h = boxes.unbind(-1) 
#     l = cx - w/2.0 
#     t = cy - h/2.0
#     r = cx + w/2.0
#     b = cy + h/2.0 

#     extent_encoding = torch.zeros((boxes.shape[0], 4, feature_width, feature_height), device=boxes.device, requires_grad=False) 

#     for w in range(feature_width):
#         for h in range(feature_height):
#             kx = w*scale_factor + scale_factor/2.0 
#             ky = h*scale_factor + scale_factor/2.0 

#             # shapes are [B]
#             li = (kx - l)/search_window_width
#             ri = (kx - r)/search_window_width
#             ti = (ky - t)/search_window_height
#             bi = (ky - b)/search_window_height

#             extent_encoding[:, 0, w, h] = li 
#             extent_encoding[:, 1, w, h] = ri 
#             extent_encoding[:, 2, w, h] = ti 
#             extent_encoding[:, 3, w, h] = bi 

#     return extent_encoding

    @staticmethod
    def generate_target_extent_encoding( boxes, scale_factor, search_window_width, search_window_height, feature_width, feature_height):
        """
        Vectorized implementation for generating target extent encoding.
        """
        cx, cy, w, h = boxes.unbind(-1)  # [B]
        l = cx - w / 2.0
        t = cy - h / 2.0
        r = cx + w / 2.0
        b = cy + h / 2.0

        device = boxes.device
        # Create meshgrid for feature map positions
        ws = torch.arange(feature_width, device=device, dtype=boxes.dtype)
        hs = torch.arange(feature_height, device=device, dtype=boxes.dtype)
        grid_w, grid_h = torch.meshgrid(ws, hs, indexing='ij')  # [feature_width, feature_height]

        kx = grid_w * scale_factor + scale_factor / 2.0  # [feature_width, feature_height]
        ky = grid_h * scale_factor + scale_factor / 2.0  # [feature_width, feature_height]

        # Expand kx, ky to [1, feature_width, feature_height]
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)

        # Expand box coordinates to [B, 1, 1]
        l = l.unsqueeze(-1).unsqueeze(-1)
        r = r.unsqueeze(-1).unsqueeze(-1)
        t = t.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        li = (kx - l) / search_window_width
        ri = (kx - r) / search_window_width
        ti = (ky - t) / search_window_height
        bi = (ky - b) / search_window_height

        # Stack and permute to [B, 4, feature_width, feature_height]
        extent_encoding = torch.stack([li, ri, ti, bi], dim=1)
        extent_encoding = extent_encoding.requires_grad_(False)
        return extent_encoding