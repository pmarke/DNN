from blocks import *

# https://www.analyticsvidhya.com/blog/2024/10/yolov11-object-detection/

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(C3K2(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(C3K2(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(C3K2(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(C3K2(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPPF(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = C3K2(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = C3K2(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = C3K2(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = C3K2(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
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
        '''
        filters: the width of the different pyramid levels that detection is done on
        '''
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        print("f",len(filters))
    
        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        
        self.box = torch.nn.ModuleList(
           torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
           torch.nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
            Conv(x, cls, torch.nn.SiLU()),
            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
            Conv(cls, cls, torch.nn.SiLU()),
            torch.nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)


    def forward(self, x):
        # The boxes are [l,r,u,d] consisting of 16 bins each
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1) 
        if self.training:
            return x
        
        # x contains the different head outputs at 3 different pyramid levles and is thus a list of 3. 
        # Each element in x is a (4*16*80) tensor where (4*16) is for the box regression and 80 is for 
        # the classes

        print("strides", self.stride)

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        # The anchors are all of the different potential object locations in the different output levels and 
        # the strides are the corresponding scales. They have not be scaled to match the original input size yet. 
        
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        # The shape of x is (b, ch, h*w) and contains the output of all levels. 
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        # box is shape (b,4*self.ch,h*w)

        a, b = self.dfl(box).chunk(2, 1)
        # a, b contains the offests for [l,r,t,b]. The code below will get the tl and br corners of the box
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        # self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x)) 
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            if m.out_channels == 80:
                # init classification bias to low confidence
                # log(p/(1-p)) where p is small, e.g. p=0.01
                if hasattr(m, 'bias') and m.bias is not None:
                    prior_prob = 0.01
                    bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, 0)
            else:
                # He init for ReLU, or Xavier for smoother activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
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