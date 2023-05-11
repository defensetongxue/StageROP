from torchvision import models
import os 
import torch.nn as nn
def build_inception3_pretrained(config,num_classes=5):
    os.environ['TORCH_HOME']=config.RESULT_PATH
    model=models.inception_v3(pretrained=True)
    model.fc=nn.Linear(2048,num_classes)
    model.AuxLogits.fc=nn.Linear(768,num_classes)

    return model
    