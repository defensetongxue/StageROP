import torch.nn as nn
from .meta_model import build_inception3_pretrained,build_vgg16_pretrained
class Inception3(nn.Module):
    def __init__(self,configs,num_classes):
        super(Inception3,self).__init__()
        self.model_heatmap=build_inception3_pretrained(configs,num_classes)

    def forward(self,x):
        if self.training:
            x,au=self.model_heatmap(x)
            return [x,au]
        x=self.model_heatmap(x)
        return  x

class VGG16(nn.Module):
    def __init__(self,configs,num_classes):
        super(VGG16,self).__init__()
        self.model_heatmap=build_vgg16_pretrained(configs,num_classes)

    def forward(self,x):
        return self.model_heatmap(x)