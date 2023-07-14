import torch.nn as nn
import torch
from .meta_model import build_inception3_pretrained,build_vgg16_pretrained
class VGG16(nn.Module):
    def __init__(self,configs,num_classes) :
        super(VGG16,self).__init__()
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
    
class Inception3(nn.Module):
    def __init__(self,configs,num_classes):
        super(Inception3,self).__init__()
        self.model_crop=build_vgg16_pretrained(configs,num_classes)
        self.model_heatmap=build_inception3_pretrained(configs,num_classes)
        self.classifier=nn.Linear(2*num_classes,num_classes)

    def forward(self,x):
        crop_tensor,heatmap_tensor=x
        crop_res=self.model_crop(crop_tensor)
        if self.training:
            heatmap_res,heatmap_au=self.model_heatmap(heatmap_tensor)
            x=torch.cat([crop_res,heatmap_res],dim=-1)
            x=self.classifier(x)
            return x,heatmap_au
        else:
            heatmap_res=self.model_heatmap(heatmap_tensor)
            x=torch.cat([crop_res,heatmap_res],dim=-1)
            x=self.classifier(x)
            return x
