import os
from torchvision import models
import torch.nn as nn

def build_vgg16_pretrained(config, num_classes=5):
    os.environ['TORCH_HOME'] = config.RESULT_PATH
    model = models.vgg16(pretrained=True)
    # VGG16 has 4096 out_features in its last Linear layer
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model
