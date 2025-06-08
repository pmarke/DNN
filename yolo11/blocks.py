import torch
import torch.nn as nn
from typing import Tuple 
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# a lot of this code is based on https://github.com/ultralytics/
# https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/
# https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71



class ConvBlock(nn.Module) :

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, group =1):
        super(ConvBlock, self).__init__()

        # With convolutional layers, Batch normalization is applied to each channel. This means that there are separate (mean, std) parameters for each 
        # channel. The idea is that the same feature map is convolved with the input to produce a single channel, so the outputs in a channel should have 
        # similar information and thus similar statistics.
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=group, bias = False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.conv(x)
   
class BottleNeckBlock(nn.Module):
    def __init__(self, chin, chout, use_residual=True, k: Tuple[int,int]=(3,3), e: float = 0.5 ):
        """
        Initialize a standard bottleneck module.

        Args:
            chin (int): Input channels.
            chout (int): Output channels.
            use_residual (bool): Whether to use shortcut connection.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super(BottleNeckBlock,self).__init__() 
        c_ = int(chout*e)
        self.use_residual = use_residual and chin == chout 
        self.conv1 = ConvBlock(chin,c_,k[0],padding=1) 
        self.conv2 = ConvBlock(c_,chout,k[1],padding=1) 

    def forward(self,x):

        y = self.conv2(self.conv1(x))

        return x + y if self.use_residual else y

class C3Block(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c1, c_, 1, 1)
        self.cv3 = ConvBlock(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleNeckBlock(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3KBlock(C3Block):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    def __init__(self, c1: int,c2: int, n: int=1, use_residual=True, g: int = 1, e: float = 0.5, k:int = 3):
        """
        Initialize C3K module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            use_residual (bool): Whether to use residual connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super(C3KBlock,self).__init__(c1,c2,n,use_residual,g,e) 
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleNeckBlock(c_, c_, use_residual, g, k=(k,k), e=1.0) for _ in range(n)))

    
class C3K2Block(nn.Module):
    def __init__(
        self, c1: int, c2: int, n: int = 1, e: float = 0.5, g: int = 1, use_residual: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            C3KBlock(self.c, self.c, 2, use_residual, g) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# batch x channels x height x width    

class SppfBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    
    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class ConvMultiHeadFlashAttention(nn.Module):
    def __init__(self, ch, num_head):
        '''
        Multi-head attention with FlashAttention using convolutional QKV projection
        '''
        super().__init__()
        self.num_head = num_head 
        self.dim_head = ch // num_head
        assert ch % num_head == 0, "ch must be divisible by num_head"
        assert self.dim_head % 2 == 0, "dim_head must be divisible by 2"

        self.scale = self.dim_head ** -0.5

        chout = self.dim_head * num_head * 3
        self.qkv = nn.Conv2d(ch, chout, kernel_size=3, padding=1)


    def forward(self,x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)  # [b, chout, h, w]

        # Flatten spatial dims
        qkv = qkv.view(b, -1, h * w).transpose(1, 2)  # [b, hw, chout]

        total_dim = self.num_head * self.dim_head
        q_dim = self.num_head * self.dim_head
        k_dim = q_dim
        v_dim = total_dim

        # Split into q, k, v
        q, k, v = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)

        # Reshape for flash attention
        def reshape_qkv(x, d): return x.view(b, h*w, self.num_head, d)

        q = reshape_qkv(q, self.dim_head)
        k = reshape_qkv(k, self.dim_head)
        v = reshape_qkv(v, self.dim_head)

        qkv_packed = torch.stack([q, k, v], dim=2)  # [b, seqlen, 3, nheads, dim]

        out = flash_attn_qkvpacked_func(qkv_packed, dropout_p=0.0, softmax_scale=self.scale, causal=False)

        out = out.view(b, h*w, -1).transpose(1, 2).view(b, c, h, w)

        return out


# class C2PSA(nn.Module):


# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
#   s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
#   m: [0.50, 1.00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
#   l: [1.00, 1.00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
#   x: [1.00, 1.50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# # YOLO11n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 2, C3k2, [256, False, 0.25]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 2, C3k2, [512, False, 0.25]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 2, C3k2, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 2, C3k2, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
#   - [-1, 2, C2PSA, [1024]] # 10

# # YOLO11n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 2, C3k2, [512, False]] # 13

#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 13], 1, Concat, [1]] # cat head P4
#   - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 10], 1, Concat, [1]] # cat head P5
#   - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

#   - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)