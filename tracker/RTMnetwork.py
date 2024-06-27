import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class RTMNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 x 128^2
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64 x 64^2
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 128 x 64^2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 128 x 32^2
        
        self.e51 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # 256 x 32^2
        
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 128 x 64^2
        # cat 256 x 64^2
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 128 x 64^2
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 64 x 128^2
        # cat 128 x 128^2
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 64 x 128^2
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 32 x 256^2
        # cat 64 x 256^2
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 32 x 256^2
        
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1) # 3 x 256^2

        
    def forward(self, x):
        xe11 = relu(self.e11(x))
        xe21 = relu(self.e21(self.pool1(xe11)))
        xe31 = relu(self.e31(self.pool2(xe21)))
        xe51 = relu(self.e51(self.pool3(xe31)))
        xu1 = self.upconv1(xe51)
        xu11 = torch.cat([xu1, xe31], dim=1)
        xd11 = relu(self.d11(xu11))
        xu2 = self.upconv2(xd11)
        xu22 = torch.cat([xu2, xe21], dim=1)
        xd21 = relu(self.d21(xu22))
        xu3 = self.upconv3(xd21)
        xu33 = torch.cat([xu3, xe11], dim=1)
        xd31 = relu(self.d31(xu33))
        out = self.outconv(xd31)
        y = x - out
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.ones_(m.weight.data)
                nn.init.normal_(m.bias.data, std= 0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()