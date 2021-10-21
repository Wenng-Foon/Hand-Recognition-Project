import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import math

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEModule, self).__init__()
        
        #self.globalmaxpool = nn.MaxPool2d(math.floor(in_channels/reduction_ratio),1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sefc1 = nn.Linear(in_channels, math.floor(in_channels/reduction_ratio))
        self.relu = nn.ReLU()
        self.sefc2 = nn.Linear(math.floor(in_channels/reduction_ratio), in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, num_channels, h,w = x.size()
        
        se = self.avg_pool(x)
        se = torch.flatten(se,1)
        se= self.relu(self.sefc1(se))
        se= self.sigmoid(self.sefc2(se))
        se = se.view(batch_size,num_channels,1,1)
        
        x_hat = x * se
            
        return x_hat

class SE_Layer(nn.Module):
    def __init__(self, bottleneck_layers, reduction_ratio):
        super(SE_Layer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.se_bottlenecks = nn.ModuleList([self.make_se_copy(l) for l in bottleneck_layers])
        
    def make_se_copy(self, bottleneck_layer):                           
        return SE_Bottleneck(bottleneck_layer, self.reduction_ratio)                        
                                         
    def forward(self, x):

        for se_bottleneck in self.se_bottlenecks:
            x = se_bottleneck(x)                 
        return x
                                            
class SE_Bottleneck(nn.Module):
    def __init__(self, bottleneck_layer, reduction_ratio):
        super(SE_Bottleneck, self).__init__()
        self.conv1 = bottleneck_layer.conv1
        self.bn1 = bottleneck_layer.bn1
        self.conv2 = bottleneck_layer.conv2
        self.bn2 = bottleneck_layer.bn2
        self.conv3 = bottleneck_layer.conv3
        self.bn3 = bottleneck_layer.bn3
        self.relu = bottleneck_layer.relu
        self.se = SEModule(self.bn3.num_features, reduction_ratio)
        if bottleneck_layer.downsample:
            self.downsample = bottleneck_layer.downsample
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
            
        return out
        
class SE_ResNet50(nn.Module):
    def __init__(self, pretrained=True, reduction_ratio = 16):
        super(SE_ResNet50, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.pretrained = pretrained
        
        #transfer leaning
        model = models.resnet50(pretrained=self.pretrained)
        
        self.backbone = nn.Sequential(
                        model.conv1,
                        model.bn1,
                        model.relu,
                        model.maxpool
                    )
        self.layer1 = SE_Layer(model.layer1, reduction_ratio)
        self.layer2 = SE_Layer(model.layer2, reduction_ratio)
        self.layer3 = SE_Layer(model.layer3, reduction_ratio)
        self.layer4 = SE_Layer(model.layer4, reduction_ratio)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 10)

    def forward(self, x): 
        
        x =self.backbone(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
