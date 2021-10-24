import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import math

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio, pool_type="avg"):
        super(SEModule, self).__init__()
        
        self.pool_type = pool_type
        
        if self.pool_type == "both":
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.pool_type ==  "max":
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
            
        self.se_bottleneck = nn.Sequential(
            nn.Linear(in_channels, math.floor(in_channels/reduction_ratio)),
            nn.ReLU(),
            nn.Linear(math.floor(in_channels/reduction_ratio), in_channels),
            )
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        
        if self.pool_type == "both":
            se1 = self.avg_pool(x)
            se1 = torch.flatten(se1,1)
            se1 = self.se_bottleneck(se1)
            
            se2 = self.max_pool(x)
            se2 = torch.flatten(se2,1)
            se2 = self.se_bottleneck(se2)
            
            se = se1+se2
            
        elif self.pool_type == "max":
            
            se = self.max_pool(x)
            se = torch.flatten(se,1)
            se = self.se_bottleneck(se)
            
        else: 
            se = self.avg_pool(x)
            se = torch.flatten(se,1)
            se = self.se_bottleneck(se)

        se= self.sigmoid(se)
        se = se.view(batch_size,num_channels,1,1)
        
        x_hat = x * se
            
        return x_hat

class SpatialModule(nn.Module):
    def __init__(self):
        super(SpatialModule, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, 7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
    
        #channel-wise pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        #concatenating pooled features and extracting spatial weights
        cat_pool = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv1(cat_pool)
        y = self.bn(y)
        y = self.sigmoid(y)
    
        x_hat = x * y
    
        return x_hat
    
class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio, pool_type, use_Spatial = False):
        super(ChannelSpatialAttention, self).__init__()
        
        self.use_Spatial = use_Spatial
        if self.use_Spatial:
            self.channel = SEModule(in_channels, reduction_ratio, pool_type = "both")
            self.spatial = SpatialModule()
        else:
            self.channel = SEModule(in_channels, reduction_ratio, pool_type = pool_type)

    def forward(self, x):
    
        #channel_attention
        x = self.channel(x)
        
        if self.use_Spatial:
            x = self.spatial(x)

        return x
    
class Attention_Layer(nn.Module):
    def __init__(self, bottleneck_layers, reduction_ratio, pool_type, use_Spatial):
        super(Attention_Layer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.pool_type = pool_type
        self.use_Spatial = use_Spatial
        self.attention_bottlenecks = nn.ModuleList([self.make_attention_copy(l) for l in bottleneck_layers])

        
    def make_attention_copy(self, bottleneck_layer):    
        return Attention_Bottleneck(bottleneck_layer, self.reduction_ratio, self.pool_type, self.use_Spatial)                        
                                         
    def forward(self, x):

        for bottleneck in self.attention_bottlenecks:
            x = bottleneck(x)                 
        return x
                                            
class Attention_Bottleneck(nn.Module):
    def __init__(self, bottleneck_layer, reduction_ratio, pool_type, use_Spatial):
        super(Attention_Bottleneck, self).__init__()
        self.conv1 = bottleneck_layer.conv1
        self.bn1 = bottleneck_layer.bn1
        self.conv2 = bottleneck_layer.conv2
        self.bn2 = bottleneck_layer.bn2
        self.conv3 = bottleneck_layer.conv3
        self.bn3 = bottleneck_layer.bn3
        self.relu = bottleneck_layer.relu
        self.attention = ChannelSpatialAttention(self.bn3.num_features, reduction_ratio, pool_type=pool_type, use_Spatial=use_Spatial)
        self.downsample = None
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
        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
            
        return out
        
class Attention_ResNet50(nn.Module):
    def __init__(self, pretrained=True, reduction_ratio = 16, pool_type="avg", use_Spatial= False):
        
        """
        mode types : pool_type: "avg" or "max" pooling method for the SE module, uses both for CBAM
                     Use_Spatial: False: standard squeeze excitement (channel attention)
                                  True: "CBAM" Channel and Spatial attention
        """
    
        super(Attention_ResNet50, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.pretrained = pretrained
        self.pool_type = pool_type
        self.use_Spatial = use_Spatial
        
        #transfer leaning
        model = models.resnet50(pretrained=self.pretrained)
        
        self.backbone = nn.Sequential(
                        model.conv1,
                        model.bn1,
                        model.relu,
                        model.maxpool
                    )
        self.layer1 = Attention_Layer(model.layer1, self.reduction_ratio, self.pool_type, self.use_Spatial)
        self.layer2 = Attention_Layer(model.layer2, self.reduction_ratio, self.pool_type, self.use_Spatial)
        self.layer3 = Attention_Layer(model.layer3, self.reduction_ratio, self.pool_type, self.use_Spatial)
        self.layer4 = Attention_Layer(model.layer4, self.reduction_ratio, self.pool_type, self.use_Spatial)
        
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