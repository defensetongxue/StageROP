from typing import Any
import torch.utils.data as data
from PIL import Image,ImageEnhance  
import os
import os.path
import torch
from torchvision import transforms
import json
class both_Dataset(data.Dataset):
    '''
        └───data
            │
            └───'crop_ridge_images'
            │   │
            │   └───001.jpg
            │   └───002.jpg
            │   └───...
            │
            └───'crop_ridge_annotations'
                │
                └───train.json
                └───valid.json
                └───test.json
    '''
    def __init__(self, data_path,split='train',resize=(800,800)):

        
        self.annotations = json.load(open(os.path.join(data_path, 
                                                       'crop_ridge_annotations', f"{split}.json")))
        self.heatmap_dict=os.path.join(data_path,'ridge_heatmap')

        if split=="train" or split== "augument":
            self.img_transform=transforms.Compose([
                ContrastEnhancement(),
                # transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])
                # the mean and std is calculate by rop1 13 samples
                ])
            self.heatmap_transform=transforms.Compose([
                    TensorhortonFlip(),
                    TensorVerticalFlip(),
                    TensorRotate(),
                    TensorNorm()
                ])
        elif split=='val' or split=='test':
            self.img_transform=transforms.Compose([
                ContrastEnhancement(),
                # transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])
            ])
            self.heatmap_transform=transforms.Compose([
                TensorNorm()
            ])
        else:
            raise ValueError(f"ilegal spilt : {split}")
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        '''
        The json format is
        ({
                "image_path":...,
                "crop_from":...,
                "class":...
            })
        '''
        # Load the image and label
        annotation = self.annotations[idx]
        image_path= annotation['image_path']
        img=Image.open(image_path)
        image_name= os.path.basename(annotation['crop_from'])
        heatmap=torch.load(os.path.join(self.heatmap_dict,image_name.split('.')[0]+'.pt'))
        label=annotation['class']

        # Transforms the image
        img=self.img_transform(img)
        heatmap=self.heatmap_transform(heatmap)
        # Store esscencial data for visualization (Gram)
        meta={}
        meta['image_path']=image_path

        return (img,heatmap),label,meta
    
    def num_classes(self):
        unique_classes = set(annot['class'] for annot in self.annotations)
        return len(unique_classes)

class Fix_RandomRotation:
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def get_params(self):
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        img = transforms.functional.rotate(
            img, angle, transforms.functional.InterpolationMode.NEAREST, 
            expand=self.expand, center=self.center)
        return img
    

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img
    
class TensorVerticalFlip():
    def __call__(self,input_tensor):
        p = torch.rand(1)
        if p >0.5:
            input_tensor=torch.flip(input_tensor,dims=[1])
        return input_tensor
class TensorhortonFlip():
    def __call__(self,input_tensor):
        p = torch.rand(1)
        if p >0.5:
            input_tensor=torch.flip(input_tensor,dims=[0])
        return input_tensor
class TensorRotate():
    def __call__(self,input_tensor):
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            input_tensor=torch.rot90(input_tensor,-1)
        elif p >= 0.25 and p < 0.5:
            input_tensor=torch.rot90(input_tensor,1)
        elif p >= 0.5 and p < 0.75:
            input_tensor=torch.rot90(input_tensor,2)
        return input_tensor
class TensorNorm():
    def __call__(self, input_tensor):
        input_tensor=input_tensor.unsqueeze(0)
        input_tensor=input_tensor.repeat(3,1,1)
        return input_tensor