from imports import *

def basicblock_mobilenet(in_, 
               out_, 
               kernel_size=3, 
               stride=1, 
               groups=1):

    
    padding = (kernel_size - 1) // 2
    block = nn.Sequential(nn.Conv2d(in_, out_, kernel_size, stride, padding, groups=groups, bias=False),
                          nn.BatchNorm2d(out_),
                          nn.ReLU6(inplace=True))
    
    return block


class InvertedResblock(nn.Module):
    
    def __init__(self, in_, out_, kernel_size=3, stride=1, expand_ratio=1, basicblock=basicblock_mobilenet):
        super(InvertedResblock, self).__init__()
        
        self.stride = stride
        hidden_dim = int(round(in_ * expand_ratio))
        blocks = []
        self.use_res_connect = self.stride == 1 and in_ == out_
        
        if expand_ratio != 1:
            blocks.append(basicblock(in_, hidden_dim, kernel_size=1))

        blocks.append(basicblock(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim))
        blocks.append(nn.Conv2d(hidden_dim, out_, kernel_size=1, stride=1, bias=False))
        blocks.append(nn.BatchNorm2d(out_))

        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.blocks(x)
        else:
            return self.blocks(x)
        

## Mobilenet with width and depth multipliers as input
class MobileNet(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0, depth_mult=None):
        super(MobileNet, self).__init__()
        block = InvertedResblock
        basicblock = basicblock_mobilenet
        input_channel = 32
        last_channel = 1280
        
        inverted_residual_setting1 = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1]
        ]
        inverted_residual_setting2 = [
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        input_channel = int(input_channel * width_mult)
        self.last_channel = last_channel
        features1 = [basicblock(3, input_channel, 3, 2, 1)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting1:
            output_channel = int(c * width_mult)
            if depth_mult: n = int(np.ceil(n*depth_mult))
            for i in range(n):
                stride = s if i == 0 else 1
                features1.append(block(input_channel, output_channel, 3, stride, expand_ratio=t))
                input_channel = output_channel
        self.features1 = nn.Sequential(*features1)
        
        
        features2 = []
        # building last several layers
        for t, c, n, s in inverted_residual_setting2:
            output_channel = int(c * width_mult)
            if depth_mult: n = int(np.ceil(n*depth_mult))
            for i in range(n):
                stride = s if i == 0 else 1
                features2.append(block(input_channel, output_channel, 3, stride, expand_ratio=t))
                input_channel = output_channel
        
        features2.append(basicblock(input_channel, self.last_channel, kernel_size=1))
        self.features2 = nn.Sequential(*features2)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    

def basicblock_resnet(in_, out_, kernel_size=3, stride=1, dilation=1, groups=1):
    padding = kernel_size//2
    block = nn.Sequential(nn.Conv2d(in_, out_, kernel_size, stride, padding, groups=groups, bias=False),
                          nn.BatchNorm2d(out_))    
    return block


class mainblock(nn.Module): 
    def __init__(self, in_, out_, basicblock=basicblock_resnet):  
        super(mainblock, self).__init__()
        self.apply_shortcut = not (in_ == out_)
        stride = 1
        if self.apply_shortcut:
            self.shortcut = basicblock(in_, out_, 1, 2)
            stride = 2
 
        self.layers = nn.Sequential(basicblock(in_, out_, 3, stride),
                                    nn.ReLU(inplace=True),
                                    basicblock(out_, out_, 3, 1))

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):     
        if self.apply_shortcut: 
            return self.relu(self.shortcut(x) + self.layers(x))
        else: 
            return self.relu(x + self.layers(x))
        
class depthwise_block(nn.Module):
    
    def __init__(self, in_, out_, activation=nn.ReLU, basicblock=basicblock_resnet):
        
        super(depthwise_block, self).__init__()
        self.apply_shortcut = not (in_ == out_)
        stride = 1
        if self.apply_shortcut:
            self.shortcut = basicblock(in_, out_, 1, 2)
            stride = 2 
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(basicblock(in_, in_, 3, stride, groups=in_),
                                    activation(inplace=True),
                                    basicblock(in_, out_, 1))
        
    def forward(self, x):
        if self.apply_shortcut: 
            return self.relu(self.shortcut(x) + self.layers(x))
        else: 
            return self.relu(x + self.layers(x))
        

class resnet18(nn.Module): 
    def __init__(self, block=mainblock, num_classes=1, width_mult=1.0, 
                 basicblock=basicblock_resnet,
                 inverted_residual_setting1=None, 
                 inverted_residual_setting2=None):
        
        super(resnet18, self).__init__()
        
        input_channel = 64
        last_channel = 512
        
        if inverted_residual_setting1 is None:
            inverted_residual_setting1 = [[64, 2], [128, 2]]
        
        if inverted_residual_setting2 is None:
            inverted_residual_setting2 = [[256, 2], [512, 1]]
        
        input_channel = int(input_channel * width_mult)
        self.last_channel = last_channel
        features1 = [basicblock(3, input_channel, 7, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)]
        
        for c, n in inverted_residual_setting1:
            output_channel = int(c * width_mult)
            for i in range(n):
                features1.append(block(input_channel, output_channel))
                input_channel = output_channel
        self.features1 = nn.Sequential(*features1)
        
        features2 = []
        for c, n in inverted_residual_setting2:
            output_channel = int(c * width_mult)
            for i in range(n):
                features2.append(block(input_channel, output_channel))
                input_channel = output_channel
        
        features2.append(block(input_channel, last_channel))
        self.features2 = nn.Sequential(*features2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x