import torch 



class SDD(torch.nn.Module):

    def __init__():
        super(SDD,self).__init__()

        self.activation = torch.nn.ReLU(implace=False)
        self.maxpool_floor = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,return_indices=False,ceil_mode=False)
        self.maxpool_ceil = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,return_indices=False,ceil_mode=True)
        self.maxpool_33 = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=0,dilation=1,return_indices=False,ceil_mode=True)

        # torch.nn.Conv2d(in_channels=3, out_channels = , kernel_size = 3, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        # torch.nn.ReLU(implace=False)
        # torch.nn.MaxPool2d(kernel_size=2,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=False)
        self.layer_1_1 = torch.nn.Conv2d(in_channels= 3, out_channels =64 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_1_2 = torch.nn.Conv2d(in_channels=64, out_channels =64 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        
        self.layer_2_1 = torch.nn.Conv2d(in_channels=64, out_channels =128 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_2_2 = torch.nn.Conv2d(in_channels=128, out_channels =128 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_3_1 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_3_2 = torch.nn.Conv2d(in_channels=256, out_channels =256 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_3_3 = torch.nn.Conv2d(in_channels=256, out_channels =256 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_4_1 = torch.nn.Conv2d(in_channels=256, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_4_2 = torch.nn.Conv2d(in_channels=512, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_4_3 = torch.nn.Conv2d(in_channels=512, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_5_1 = torch.nn.Conv2d(in_channels=512, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_5_2 = torch.nn.Conv2d(in_channels=512, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_5_3 = torch.nn.Conv2d(in_channels=512, out_channels =512 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')


        self.layer_conv_FC6 = torch.nn.Conv2d(in_channels=512, out_channels =1024 , kernel_size = 3, stride=1, padding=1,dilation=6,groups=1,bias=True,padding_mode='zeros')
        self.layer_conv_FC7 = torch.nn.Conv2d(in_channels=1024, out_channels =1024 , kernel_size = 1, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')


        # Auxiliary convolution layers
        self.layer_8_1 = torch.nn.Conv2d(in_channels=1024, out_channels =256 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_8_2 = torch.nn.Conv2d(in_channels=256, out_channels =512 , kernel_size = 3, stride=2, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_9_1 = torch.nn.Conv2d(in_channels=512, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_9_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=2, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_10_1 = torch.nn.Conv2d(in_channels=256, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_10_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_11_1 = torch.nn.Conv2d(in_channels=256, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_11_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')


        # Input dim 3x300x300
    def forward(self,x: torch.Tensor):
        x = self.layer_1_1(x)
        x = self.activation(x)
        x = self.layer_1_2(x)
        x = self.activation(x)
        x = self.maxpool_floor(x)

        # input dim 64x150x150
        x = self.layer_2_1(x)
        x = self.activation(x)
        x = self.layer_2_2(x)
        x = self.activation(x)
        x = self.maxpool_floor(x)

        # Input dim 128x75x75
        x = self.layer_3_1(x)
        x = self.activation(x)
        x = self.layer_3_2(x)
        x = self.activation(x)
        x = self.layer_3_3(x)
        x = self.activation(x)
        x = self.maxpool_ceil(x)

        # Input dim 256x38x38
        x = self.layer_4_1(x)
        x = self.activation(x)
        x = self.layer_4_2(x)
        x = self.activation(x)
        x = self.layer_4_3(x)
        x = self.activation(x)
        x = self.maxpool_floor(x)

        # Input dim 512x19x19
        x = self.layer_5_1(x)
        x = self.activation(x)
        x = self.layer_5_2(x)
        x = self.activation(x)
        x = self.layer_5_3(x)
        x = self.activation(x)
        x = self.maxpool_33(x)

        # Input dim 512x19x19
        x = self.layer_conv_FC6(x)
        x = self.activation(x)

        # Input dim 1024x19x19
        x = self.layer_conv_FC7(x)
        x = self.activation(x)
        
        # Input dim 1024x19x19
        x = self.layer_8_1(x)
        x = self.activation(x)
        x = self.layer_8_2(x)
        x = self.activation(x)

        # Input 512x10x10
        x = self.layer_9_1(x)
        x = self.activation(x)
        x = self.layer_9_2(x)
        x = self.activation(x)

        # Input 256x5x5
        x = self.layer_10_1(x)
        x = self.activation(x)
        x = self.layer_10_2(x)
        x = self.activation(x)

        # Input 256x3x3
        x = self.layer_11_1(x)
        x = self.activation(x)
        x = self.layer_11_2(x)
        x = self.activation(x)

        # Input 256x1x1





