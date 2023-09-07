import torch.nn as nn
import torch
from .meta_model import build_inception3_pretrained,build_vgg16_pretrained
class VGG16(nn.Module):
    def __init__(self,configs,num_classes) :
        super(VGG16,self).__init__()
        self.model_img_crop=build_vgg16_pretrained(configs,num_classes)
        self.model_vessel_crop=build_vgg16_pretrained(configs,num_classes)
        self.model_heatmap=build_vgg16_pretrained(configs,num_classes)
        self.classifier=nn.Linear(3*num_classes,num_classes)
    def forward(self,x):
        img_crop_tensor,vessel_crop_tensor,heatmap_tensor=x
        crop_res=self.model_img_crop(img_crop_tensor)
        vessel_res=self.model_vessel_crop(vessel_crop_tensor)
        heatmap_res=self.model_heatmap(heatmap_tensor)
        x=torch.cat([crop_res,vessel_res,heatmap_res],dim=-1)
        x=self.classifier(x)
        return x
    
class Inception3(nn.Module):
    def __init__(self,configs,num_classes):
        super(Inception3,self).__init__()
        self.model_img_crop=build_inception3_pretrained(configs,num_classes)
        self.model_vessel_crop=build_inception3_pretrained(configs,num_classes)
        self.model_heatmap=build_inception3_pretrained(configs,num_classes)
        self.classifier=nn.Linear(3*num_classes,num_classes)
    def forward(self,x):
        img_crop_tensor,vessel_crop_tensor,heatmap_tensor=x
        if self.training:
            crop_res,crop_au=self.model_img_crop(img_crop_tensor)
            vessel_res,vessel_au=self.model_vessel_crop(vessel_crop_tensor)
            heatmap_res,heatmap_au=self.model_heatmap(heatmap_tensor)
            x=torch.cat([crop_res,vessel_res,heatmap_res],dim=-1)
            x=self.classifier(x)
            return x,crop_au,vessel_au,heatmap_au

        else:
            crop_res=self.model_img_crop(img_crop_tensor)
            vessel_res=self.model_vessel_crop(vessel_crop_tensor)
            heatmap_res=self.model_heatmap(heatmap_tensor)
            x=torch.cat([crop_res,vessel_res,heatmap_res],dim=-1)
            x=self.classifier(x)
            return x
