import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SEModule(nn.Module):
    def __init__(self, in_channels, r_val):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        #self.globalmaxpool = nn.MaxPool2d(in_channels/r_val,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sefc1 = nn.Linear(in_channels, in_channels/r_val)
        #self.batch1= nn.BatchNorm1d(in_channels/r_val)
        self.sefc2 = nn.Linear(in_channels/r_val, in_channels )
        #self.batch2= nn.BatchNorm1d(in_channels)
        
    def forward(self, x):

        se = self.avg_pool(x)
        se = torch.flatten(se,1)
        se= F.relu(self.sefc1(se))
        se= torch.sigmoid(self.sefc2(se))
        se = se.view(batch_size,64,1,1)
        x_hat = x*se
            
        return x_hat
            
class RecognitionModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10
        self.r_val = 8

        model = models.resnet18(pretrained=True)
        
        # TODO: make this resnet 50 with custom res blocks to incorporate SE module
        self.backbone = nn.Sequential(
                        model.conv1,
                        model.bn1,
                        model.relu,
                        model.maxpool,
                        model.layer1
                    )
        self.se1 = SEModule(64, r_val)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        
        #batch_size = imgs.size()[0]
        
        x = self.backbone(x)
        x = self.se1(x)
        
        cls_scores = self.fc1(x)

    return cls_scores