"""Small residual CNN and fixed, template-compatible weapon ROI; CPU only."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from weapon_lamp_detect.region_matcher import weapon_region

HEIGHT, WIDTH = 64,96


def image_input(crop):
    region=weapon_region(crop,False)  # Existing canonical canvas / badge exclusion.
    return cv2.cvtColor(cv2.resize(region,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()


def augment(rgb):
    """Train-only gentle pose/colour/blur/flash augmentation, no test adaptation."""
    n=rgb.shape[0];device=rgb.device
    angle=(torch.rand(n,device=device)*2-1)*.0872665
    scale=.92+.16*torch.rand(n,device=device)
    flip=torch.where(torch.rand(n,device=device)<.5,-1.,1.)
    theta=torch.zeros(n,2,3,device=device)
    theta[:,0,0]=angle.cos()*scale*flip;theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale*flip;theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=(torch.rand(n,2,device=device)*2-1)*.055
    rgb=F.grid_sample(rgb,F.affine_grid(theta,rgb.shape,align_corners=False),padding_mode='border',align_corners=False)
    mean=rgb.mean(dim=1,keepdim=True)
    saturation=.8+.4*torch.rand(n,1,1,1,device=device)
    gain=.8+.4*torch.rand(n,1,1,1,device=device)
    channels=.95+.1*torch.rand(n,3,1,1,device=device)
    rgb=(mean+(rgb-mean)*saturation)*gain*channels
    blur=(torch.rand(n,1,1,1,device=device)<.12).float()
    rgb=rgb*(1-blur)+F.avg_pool2d(rgb,3,stride=1,padding=1)*blur
    flash=(torch.rand(n,1,1,1,device=device)<.1).float()*torch.rand(n,1,1,1,device=device)*.2
    return (rgb*(1-flash)+flash).clamp(0,1)


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(incoming,outgoing,3,stride=stride,padding=1,bias=False),
                                nn.BatchNorm2d(outgoing),nn.ReLU(),
                                nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),nn.BatchNorm2d(outgoing))
        self.skip=(nn.Identity() if incoming==outgoing and stride==1 else
                   nn.Sequential(nn.Conv2d(incoming,outgoing,1,stride=stride,bias=False),nn.BatchNorm2d(outgoing)))

    def forward(self,x):return F.relu(self.body(x)+self.skip(x))


class WeaponCNN(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(4,16,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(16),nn.ReLU(),
                                    Residual(16,16,1),Residual(16,32,2),Residual(32,64,2),Residual(64,96,2),
                                    nn.AdaptiveAvgPool2d((2,3)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(96*6,256),nn.ReLU(),nn.Dropout(.15),nn.Linear(256,classes))

    def forward(self,rgb):
        gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        local=(gray-F.avg_pool2d(gray,7,stride=1,padding=3)).clamp(-.5,.5)*2
        return self.head(self.features(torch.cat((rgb*2-1,local),dim=1)))


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='weapon_residual_v1' or checkpoint.get('input_size')!=[HEIGHT,WIDTH]:
        raise ValueError('未対応のCNN checkpointです')
    model=WeaponCNN(len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model,checkpoint
