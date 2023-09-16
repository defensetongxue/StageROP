from torchvision import models
import os 
import torch.nn as nn
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def build_inceptionv3(config):
    os.environ['TORCH_HOME']=config["official_model_save"]
    model=models.inception_v3(pretrained=True)
    model.fc=nn.Linear(2048,config["num_classes"])
    model.AuxLogits.fc=nn.Linear(768,config.num_classes)

    return model
def build_vgg16(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.vgg16(pretrained=True)
    # VGG16 has 4096 out_features in its last Linear layer
    model.classifier[6] = nn.Linear(4096,config["num_classes"])

    return model
    
def build_mobelnetv3_large(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.mobilenet_v3_large(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1280, out_features=config["num_classes"], bias=True)
    print(f"mobile net v3 large has {count_parameters(model)}")
    return model
def build_mobelnetv3_small(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.mobilenet_v3_small(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1024, out_features=config["num_classes"], bias=True)
    print(f"mobile net v3 large has {count_parameters(model)}")
    return model
def build_efficientnet_b7(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.efficientnet_b7(pretrained=True)
    model.classifier[1]=nn.Linear(in_features=2560, out_features=config['num_classes'], bias=True)
    print(f"efficentnet b7 has {count_parameters(model)}")
    # return model
if __name__ =='__main__':
    cfg={
        "official_model_save":"./experiments",
        "num_classes":2
    }
    build_efficientnet_b7(cfg)