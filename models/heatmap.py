import torch.nn as nn
import torch
from .meta_model import build_inception3_pretrained,build_vgg16_pretrained
class Inception_v3_heatmap(nn.Module):
    def __init__(self,configs,num_classes):
        super(Inception_v3_heatmap,self).__init__()
        self.model_heatmap=build_inception3_pretrained(configs,num_classes)

    def forward(self,x):
        return self.model_heatmap(x)


class vgg_heatmap(nn.Module):
    def __init__(self,configs,num_classes):
        super(vgg_heatmap,self).__init__()
        self.model_heatmap=build_vgg16_pretrained(configs,num_classes)

    def forward(self,x):
        return self.model_heatmap(x)