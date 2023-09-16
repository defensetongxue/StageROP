import torch
from . import meta_model
from torch import nn
class mutil_model(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        self.model0=getattr(meta_model,f"build_{configs['name'][0]}")(configs['model0'])
        self.model1=getattr(meta_model,f"build_{configs['name'][1]}")(configs['model1'])
    def forward(self,x):
        x0,x1=x
        out0=self.model0(x0)
        out1=self.model1(x1)
        return (out0,out1)
class multi_loss(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        self.loss0=getattr(nn,configs["loss_func"][0])()
        self.loss1=getattr(nn,configs["loss_func"][1])()
        self.r=configs['loss_r0']
    def forward(self,inputs,targets):
        in0,in1=inputs
        tar0,tar1=targets
        return self.r*self.loss0(in0,tar0)+(1-self.r)*self.loss1(in1,tar1)
    
def build_model(configs):
    if isinstance(configs['name'],list):
        model= mutil_model(configs)
        loss_func=multi_loss(configs)
        return model,loss_func
    else:
        model= getattr(meta_model,f"build_{configs['name']}")(configs)
        loss_func=getattr(nn,configs["loss_func"])()
        return model,loss_func