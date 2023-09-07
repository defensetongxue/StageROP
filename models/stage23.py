import torch.nn as nn
import torch
class fc(nn.Module):
    def __init__(self,configs,num_classes):
        super(fc,self).__init__()
        self.hidden=nn.Linear(5,32)
        self.classifier=nn.Linear(32,num_classes)
        self.drop=nn.Dropout()
        self.act=nn.ReLU()
    def forward(self,x):
        x=self.hidden(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.classifier(x)
        return x