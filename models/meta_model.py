from torchvision import models
import os 
import torch.nn as nn
import torch

def build_inception3_pretrained(config,num_classes=5):
    os.environ['TORCH_HOME']=config.RESULT_PATH
    model=models.inception_v3(pretrained=True)
    model.fc=nn.Linear(2048,num_classes)
    model.AuxLogits.fc=nn.Linear(768,num_classes)

    return model

def build_vgg16_pretrained(config, num_classes=5):
    os.environ['TORCH_HOME'] = config.RESULT_PATH
    model = models.vgg16(pretrained=True)
    # VGG16 has 4096 out_features in its last Linear layer
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model
    