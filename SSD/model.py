import torch 



class SDD(torch.nn.Module):

    def __init__(self,num_classes : int):
        super(SDD,self).__init__()

        self.num_classes = num_classes
        self.total_num_classes = num_classes+1 # the one represents background
        self.loc_size = 4
        self.num_priors_4_3 = 4
        self.num_priors_7 = 6
        self.num_priors_8_2 = 6
        self.num_priors_9_2 = 6
        self.num_priors_10_2 = 4
        self.num_priors_11_2 = 4

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


        self.layer_6 = torch.nn.Conv2d(in_channels=512, out_channels =1024 , kernel_size = 3, stride=1, padding=1,dilation=6,groups=1,bias=True,padding_mode='zeros')
        self.layer_7 = torch.nn.Conv2d(in_channels=1024, out_channels =1024 , kernel_size = 1, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')


        # Auxiliary convolution layers
        self.layer_8_1 = torch.nn.Conv2d(in_channels=1024, out_channels =256 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_8_2 = torch.nn.Conv2d(in_channels=256, out_channels =512 , kernel_size = 3, stride=2, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_9_1 = torch.nn.Conv2d(in_channels=512, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_9_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=2, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_10_1 = torch.nn.Conv2d(in_channels=256, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_10_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')

        self.layer_11_1 = torch.nn.Conv2d(in_channels=256, out_channels =128 , kernel_size = 1, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_11_2 = torch.nn.Conv2d(in_channels=128, out_channels =256 , kernel_size = 3, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')

        # Location Convolutions
        self.layer_loc_4_3  = torch.nn.Conv2d(in_channels=self.layer_4_3. out_channels, out_channels =self.loc_size*self.num_priors_4_3,  kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_loc_7    = torch.nn.Conv2d(in_channels=self.layer_7.   out_channels, out_channels =self.loc_size*self.num_priors_7,    kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_loc_8_2  = torch.nn.Conv2d(in_channels=self.layer_8_2. out_channels, out_channels =self.loc_size*self.num_priors_8_2,  kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_loc_9_2  = torch.nn.Conv2d(in_channels=self.layer_9_2. out_channels, out_channels =self.loc_size*self.num_priors_9_2,  kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_loc_10_2 = torch.nn.Conv2d(in_channels=self.layer_10_2.out_channels, out_channels =self.loc_size*self.num_priors_10_2, kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_loc_11_2 = torch.nn.Conv2d(in_channels=self.layer_11_2.out_channels, out_channels =self.loc_size*self.num_priors_11_2, kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')

        # Classification Convolutions
        self.layer_classify_4_3  = torch.nn.Conv2d(in_channels=self.layer_4_3.out_channels,  out_channels =self.total_num_classes*self.num_priors_4_3  , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_classify_7    = torch.nn.Conv2d(in_channels=self.layer_7.out_channels,    out_channels =self.total_num_classes*self.num_priors_7    , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_classify_8_2  = torch.nn.Conv2d(in_channels=self.layer_8_2.out_channels,  out_channels =self.total_num_classes*self.num_priors_8_2  , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_classify_9_2  = torch.nn.Conv2d(in_channels=self.layer_9_2.out_channels,  out_channels =self.total_num_classes*self.num_priors_9_2  , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_classify_10_2 = torch.nn.Conv2d(in_channels=self.layer_10_2.out_channels, out_channels =self.total_num_classes*self.num_priors_10_2 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        self.layer_classify_11_2 = torch.nn.Conv2d(in_channels=self.layer_11_2.out_channels, out_channels =self.total_num_classes*self.num_priors_11_2 , kernel_size = 3, stride=1, padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
        
        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = torch.nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        torch.nn.init.constant_(self.rescale_factors, 20)


        # Input dim 3x300x300 RGB format
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
        x_4_3 = self.activation(x)
        x = self.maxpool_floor(x_4_3)

        # rescalse x_4_3
        norm = x_4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        x_4_3 = x_4_3 / norm  # (N, 512, 38, 38)
        x_4_3 = x_4_3 * self.rescale_factors  # (N, 512, 38, 38)

        # Input dim 512x19x19
        x = self.layer_5_1(x)
        x = self.activation(x)
        x = self.layer_5_2(x)
        x = self.activation(x)
        x = self.layer_5_3(x)
        x = self.activation(x)
        x = self.maxpool_33(x)

        # Input dim 512x19x19
        x = self.layer_6(x)
        x = self.activation(x)

        # Input dim 1024x19x19
        x = self.layer_7(x)
        x_7 = self.activation(x)
        
        # Input dim 1024x19x19
        x = self.layer_8_1(x_7)
        x = self.activation(x)
        x = self.layer_8_2(x)
        x_8_2 = self.activation(x)

        # Input 512x10x10
        x = self.layer_9_1(x_8_2)
        x = self.activation(x)
        x = self.layer_9_2(x)
        x_9_2 = self.activation(x)

        # Input 256x5x5
        x = self.layer_10_1(x_9_2)
        x = self.activation(x)
        x = self.layer_10_2(x)
        x_10_2 = self.activation(x)

        # Input 256x3x3
        x = self.layer_11_1(x_10_2)
        x = self.activation(x)
        x = self.layer_11_2(x)
        x_11_2 = self.activation(x)
        # Output 256x1x1


        # Location Layers
        loc_4_3  = self.layer_loc_4_3(x_4_3)  
        loc_7    = self.layer_loc_7(x_7)  
        loc_8_2  = self.layer_loc_8_2(x_8_2)
        loc_9_2  = self.layer_loc_9_2(x_9_2)
        loc_10_2 = self.layer_loc_10_2(x_10_2)
        loc_11_2 = self.layer_loc_11_2(x_11_2)

        # Classification layers
        classify_4_3  = self.layer_classify_4_3(x_4_3)  
        classify_7    = self.layer_classify_7(x_7)  
        classify_8_2  = self.layer_classify_8_2(x_8_2)
        classify_9_2  = self.layer_classify_9_2(x_9_2)
        classify_10_2 = self.layer_classify_10_2(x_10_2)
        classify_11_2 = self.layer_classify_11_2(x_11_2)

        # Reshape and stack the locations and classifications such that their shape is 8732x4 and 8731x(num_classes +1)
        batch_size = loc_4_3.size(0)
        loc_4_3  = loc_4_3.permute(0,2,3,1).contiguous()
        loc_4_3  = loc_4_3.view(batch_size,-1,4)
        loc_7    = loc_7.permute(0,2,3,1).contiguous()
        loc_7    = loc_7.view(batch_size,-1,4)
        loc_8_2  = loc_8_2.permute(0,2,3,1).contiguous()
        loc_8_2  = loc_8_2.view(batch_size,-1,4)
        loc_9_2  = loc_9_2.permute(0,2,3,1).contiguous()
        loc_9_2  = loc_9_2.view(batch_size,-1,4)
        loc_10_2 = loc_10_2.permute(0,2,3,1).contiguous()
        loc_10_2 = loc_10_2.view(batch_size,-1,4)
        loc_11_2 = loc_11_2.permute(0,2,3,1).contiguous()
        loc_11_2 = loc_11_2.view(batch_size,-1,4)

        locs = torch.cat([loc_4_3,loc_7,loc_8_2,loc_9_2,loc_10_2,loc_11_2], dim=1) # (N, 8732, 4)

        classify_4_3 = classify_4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_4_3 = classify_4_3.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        classify_7 = classify_7.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_7 = classify_7.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        classify_8_2 = classify_8_2.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_8_2 = classify_8_2.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        classify_9_2 = classify_9_2.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_9_2 = classify_9_2.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        classify_10_2 = classify_10_2.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_10_2 = classify_10_2.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
        classify_11_2 = classify_11_2.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        classify_11_2 = classify_11_2.view(batch_size, -1,self.total_num_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        classes_scores = torch.cat([classify_4_3, classify_7, classify_8_2, classify_9_2, classify_10_2, classify_11_2], dim=1)  # (N, 8732, n_classes)
        
        

        return locs, classes_scores




