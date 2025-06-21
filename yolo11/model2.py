from yolo11.blocks2 import * 



def fuse_conv(conv, norm):
    """
    Merges a trained convolution layer and a normalization layer into a single convolution layer by adjusting its weights and biases.
    This is commonly used during model optimization for deployment as it removes the need for separate normalization layers
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)
    


class Backbone(nn.Module): 
    def __init__(self,width,depth,c3k):
        super().__init__() 

        # The width is used to represent the channels and the depth
        # is used to indicate the number of C3K blocks. These parameters
        # can be used to build the 5 versions of YOLO nano, small, medium, 
        # large and extra large

        print("widths", width)
        print("depth", depth)
        print("c3k",c3k)

        # p1/2 
        # The input size is 3x640x640 
        self.p1 = torch.nn.Sequential(ConvBlock(width[0],width[1],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()))

        # p2/4
        # Shape is (width[1]x320x320)
        self.p2 = torch.nn.Sequential(ConvBlock(width[1],width[2],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()),C3K2Block(width[2],width[3],depth[0],c3k[0],e=0.25))

        # p3/8 
        # Shape is (width[3],160x160)
        self.p3 = torch.nn.Sequential(ConvBlock(width[3],width[3],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()),C3K2Block(width[3],width[4],depth[1],c3k[0],e=0.25))
        # self.p3 = torch.nn.Sequential(ConvBlock(width[3],width[3],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()),ConvBlock(width[3],width[3],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()))

        # # P4/16 
        # # Shape is (width[4],80x80)
        # p4_block.append(ConvBlock(width[4],width[4],kernel_size=3,stride=2,padding=1,activation=nn.SiLU()))
        # p4_block.append(C3K2Block(width[4],width[4],depth[2],c3k[1],e=0.5))


        # # P5/32
        # # Shape is (width[4],40,40)
        # p5_block.append(ConvBlock(width[4], width[5], kernel_size=3, stride=2, padding=1))
        # p5_block.append(C3K2Block(width[5], width[5], depth[3], c3k[1], e=0.5))
        # p5_block.append(SppfBlock(width[5], width[5]))
        # p5_block.append(PSABlock(width[5], depth[4]))

        # Shape is (width[5],20,20)

        # self.p4 = torch.nn.Sequential(*p4_block)
        # self.p5 = torch.nn.Sequential(*p5_block)

    def forward(self, x):
        print("backbone")
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input to Backbone must be a torch.Tensor")
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch_size, channels, height, width)")
        
        p1a = self.p1(x)
        p2a = self.p2(p1a)
        p3a = self.p3(p2a)
        # p4 = self.p4(p3)
        # p5 = self.p5(p4)
        # return p3, p4, p5
        return p3a
    
        # self.p5.append(ConvBlock(width[4], width[5], kernel_size=3, stride=2, padding=1))
        # self.p5.append(C3K2Block(width[5], width[5], depth[3], c3k[1], e=0.5))
        # self.p5.append(SppfBlock(width[5], width[5]))
        # self.p5.append(PSABlock(width[5], depth[4]))

        # Shape is (width[5],20,20)

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        # self.p4 = torch.nn.Sequential(*self.p4)
        # self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        print("backbone")
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input to Backbone must be a torch.Tensor")
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch_size, channels, height, width)")
        
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        # p4 = self.p4(p3)
        # p5 = self.p5(p4)
        # return p3, p4, p5
        return p3
    

class DarkFPN(nn.Module):
    """
    This is the neck 
    """
    def __init__(self,width,depth,c3k):
        super().__init__() 
        self.up = nn.Upsample(scale_factor=2) 

        self.h1 = C3K2Block(width[4] + width[5], width[4], depth[5], c3k[0], e=0.5)
        
        self.h2 = C3K2Block(width[4] + width[4], width[3], depth[5], c3k[0], e=0.5)
        self.h3 = ConvBlock(width[3], width[3], kernel_size=3, stride=2, padding=1)
        self.h4 = C3K2Block(width[3] + width[4], width[4], depth[5], c3k[0], e=0.5)
        self.h5 = ConvBlock(width[4], width[4], kernel_size=3, stride=2, padding=1)
        self.h6 = C3K2Block(width[4] + width[5], width[5], depth[5], c3k[1], e=0.5)

    def forward(self, x):
        print("neck")
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5
    


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        
        self.box = torch.nn.ModuleList(
           torch.nn.Sequential(ConvBlock(x, box, kernel_size=3, padding=1),
           ConvBlock(box, box, kernel_size=3, padding=1),
           torch.nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(ConvBlock(x, x, activation= torch.nn.SiLU(), kernel_size=3, padding=1, group=x),
            ConvBlock(x, cls,activation= torch.nn.SiLU(), kernel_size=1, stride=1,padding=0),
            ConvBlock(cls, cls, activation=torch.nn.SiLU(), kernel_size=3, padding=1, group=cls),
            ConvBlock(cls, cls, activation=torch.nn.SiLU(),kernel_size=1,stride=1,padding=0),
            torch.nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)

    def forward(self, x):
        print("Head")
        
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = Backbone(width, depth, csp)
        # self.fpn = DarkFPN(width, depth, csp)

        # img_dummy = torch.zeros(1, width[0], 256, 256)
        # self.head = Head(num_classes, (width[3], width[4], width[5]))
        # self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        # self.stride = self.head.stride
        # self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        return x
        x = self.fpn(x)

        return self.head(list(x))

    # def fuse(self):
    #     for m in self.modules():
    #         if type(m) is ConvBlock and hasattr(m, 'norm'):
    #             m.conv = fuse_conv(m.conv, m.norm)
    #             m.forward = m.fuse_forward
    #             delattr(m, 'norm')
    #     return self
    
class YOLOv11:
  def __init__(self):
    
    self.dynamic_weighting = {
      'n':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 16, 32, 64, 128, 256]},
      's':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 32, 64, 128, 256, 512]},
      'm':{'csp': [True, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 64, 128, 256, 512, 512]},
      'l':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 64, 128, 256, 512, 512]},
      'x':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 96, 192, 384, 768, 768]},
    }
  def build_model(self, version, num_classes):
    csp = self.dynamic_weighting[version]['csp']
    depth = self.dynamic_weighting[version]['depth']
    width = self.dynamic_weighting[version]['width']
    return YOLO(width, depth, csp, num_classes)