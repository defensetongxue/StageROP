import os
from torchvision import models
import torch.nn as nn
import torch
from .meta_model import build_vgg16_pretrained
class VGG16_both(nn.Module):
    def __init__(self,configs,num_classes) -> None:
        super(VGG16_both,self).__init__()
        self.model_crop=build_vgg16_pretrained(configs,num_classes)
        self.model_heatmap=build_vgg16_pretrained(configs,num_classes)
        self.classifier=nn.Linear(2*num_classes,num_classes)
    def forward(self,x):
        crop_tensor,heatmap_tensor=x
        crop_res=self.model_crop(crop_tensor)
        heatmap_res=self.model_heatmap(heatmap_tensor)
        x=torch.cat([crop_res,heatmap_res],dim=-1)
        x=self.classifier(x)
        return x
def build_vgg16(configs,num_classes,mode='both'):
    if mode=='both':
        return VGG16_both(configs,num_classes)
    elif mode=='heatmap' or mode=='crop':
        return build_vgg16_pretrained(configs,num_classes)
    else:
        raise ValueError(f'Unexpect mode for Inception {mode}')