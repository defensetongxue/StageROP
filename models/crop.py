import torch.nn as nn
import torch
from .meta_model import build_inception3_pretrained,build_vgg16_pretrained
class VGG16(nn.Module):
    def __init__(self,configs,num_classes):
        super(VGG16,self).__init__()
        self.crop_img_model=build_vgg16_pretrained(configs,num_classes)
        self.crop_vessel_model=build_vgg16_pretrained(configs,num_classes)
        self.classifier=nn.Linear(2*num_classes,num_classes)
    def forward(self,x):
        img,vessel=x
        img=self.crop_img_model(img)
        vessel=self.crop_vessel_model(vessel)
        x=torch.cat([img,vessel],dim=-1)
        x=self.classifier(x)
        return x
    
class Inception3(nn.Module):
    def __init__(self,configs,num_classes):
        super(Inception3,self).__init__()
        self.crop_img_model=build_inception3_pretrained(configs,num_classes)
        self.crop_vessel_model=build_inception3_pretrained(configs,num_classes)
        self.classifier=nn.Linear(2*num_classes,num_classes)
    def forward(self,x):
        img,vessel=x
        if self.training:
            img,img_au=self.crop_img_model(img)
            vessel,vessel_au=self.crop_vessel_model(vessel)
            x=torch.cat([img,vessel],dim=-1)
            x=self.classifier(x)
            return x,img_au,vessel_au
        else:
            img=self.crop_img_model(img)
            vessel=self.crop_vessel_model(vessel)
            x=torch.cat([img,vessel],dim=-1)
            x=self.classifier(x)
            return x