import torch
from . import meta_model
from torch import nn
class incetionV3_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls=nn.CrossEntropyLoss()
        self.loss_avg=nn.CrossEntropyLoss()
    def forward(self,inputs,target):
        if isinstance(inputs,tuple):
            out,avg=inputs
            return self.loss_cls(out,target)+self.loss_avg(avg,target)
        return self.loss_cls(inputs,target)
def build_model(configs):
    
    model= getattr(meta_model,f"build_{configs['name']}")(configs)
    if configs['name']=='inceptionv3':
        loss_func=incetionV3_loss()
    else:
        loss_func=getattr(nn,configs["loss_func"])()
    return model,loss_func