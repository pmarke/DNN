import torch
import torch.nn as nn
from typing import Tuple 
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# a lot of this code is based on https://github.com/ultralytics/
# https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/
# https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71
# https://github.com/ultralytics/ultralytics


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
   
class BottleNeckBlock(nn.Module):
    def __init__(self, chin, chout, use_residual=True, e: float = 0.5 ):
        """
        Initialize a standard bottleneck module.

        Args:
            chin (int): Input channels.
            chout (int): Output channels.
            use_residual (bool): Whether to use shortcut connection.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """

        super().__init__() 
        c_ = int(chout*e)
        self.use_residual = use_residual and chin == chout 
        self.conv1 = ConvBlock(chin,c_,kernel_size=3,padding=1) 
        self.conv2 = ConvBlock(c_,chout,kernel_size=3,padding=1) 

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
        self.bottlenecks = nn.Sequential(*(BottleNeckBlock(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.bottlenecks(self.cv1(x)), self.cv2(x)), 1))


    
class C3K2Block(nn.Module):
    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool=False, e: float=0.5, g: int=1, shortcut:bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.blocks = None
        
        if(c3k):
            # Use the c3k block
            print("this block")
            self.blocks = nn.ModuleList(
                C3Block(self.c, self.c, 2, shortcut) for _ in range(n)
            )
        else:
            # Use the bottle neck block
            self.blocks = nn.ModuleList(
                BottleNeckBlock(self.c, self.c, shortcut) for _ in range(n)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2,1))
        for m in self.blocks:
            y.append(m(y[-1]))
        # y.extend(m(y[-1]) for m in self.blocks)
        return self.cv2(torch.cat(y, dim=1))

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

# Code for the Attention Module

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

        self.conv1 = ConvBlock(chout,chout,activation= nn.Identity())


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

        return self.conv1(out)

class PSABlock(nn.Module):

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


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvBlock(c1, 2 * self.c)
        self.cv2 = ConvBlock(2 * self.c, c1)

        self.res_m = torch.nn.Sequential(*(PSABlock(c1 // 2, c1 // 128) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.res_m(b)
        return self.cv2(torch.cat((a, b), 1))

class DFL(nn.Module):
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
