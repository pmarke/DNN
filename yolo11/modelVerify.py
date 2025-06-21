from yolo11.blocks2 import * 



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,chin, chout,numHeads):
        super(MultiHeadAttentionBlock, self).__init__()

        self.convMHFA = ConvMultiHeadFlashAttention(chin,numHeads) 
        self.bn1 = nn.BatchNorm2d(chin)
        self.conv1 = nn.Conv2d(chin,4*chin,kernel_size=1,bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(4*chin,chout,kernel_size=1,bias=True)

        self.conv3 = None 
        if(chin != chout):
            self.conv3 = nn.Conv2d(chin,chout,kernel_size=1,bias=False)

    def forward(self,x):
         y1 = self.convMHFA(x) 
         y1 = x+y1
         y2 = self.bn1(y1)
         y2 = self.conv1(y2) 
         y2 = self.act(y2)
         y2 = self.conv2(y2)
         if(self.conv3 is not None):
             y1 = self.conv3(y1)
         return y1+y2




class YoloV11(nn.Module):
        def __init__(self, num_classes = 10, dropout=0.2):
            super(YoloV11, self).__init__()
            # input size is b,3,32,32
            self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False)
            self.bn1 = nn.BatchNorm2d(64) 
            self.act1 = nn.GELU()


            self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128) 
            self.act2 = nn.GELU()
            self.dropout = nn.Dropout(dropout)

            # input size is (b,128,16,16)
            self.mha1 = nn.Sequential(MultiHeadAttentionBlock(128,128,4) ,nn.BatchNorm2d(128) ,nn.GELU(), nn.Dropout(dropout)  )
            self.mha2 = nn.Sequential(MultiHeadAttentionBlock(128,32,4) ,nn.BatchNorm2d(32) ,nn.GELU(), nn.Dropout(dropout)  )
            self.mha3 = nn.Sequential(MultiHeadAttentionBlock(32,6,1) ,nn.BatchNorm2d(6) ,nn.GELU(), nn.Dropout(dropout)  )



            # at this point is is b,64,32,32
            self.fc = nn.Linear(1536, num_classes)
            self.softmax = nn.Softmax()

      
        def forward(self, x):

            x = self.conv1(x)
            x = self.bn1(x) 
            x = self.act1(x) 
            x = self.dropout(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.dropout(x)

            x = self.mha1(x)
            x = self.mha2(x)
            x = self.mha3(x)
            b,c,h,w  = x.shape
            x = x.view(b,-1)
            x = self.fc(x)
            x = self.softmax(x)

            return x
