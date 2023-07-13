import torch.utils.data as data
from PIL import Image,ImageEnhance  
import torch.nn.functional as F
import os
import os.path
import torch
from torchvision import transforms
import json
class heatmap_Dataset(data.Dataset):
    '''
        └───data
            │
            └───'heatmap'
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
    def __init__(self, data_path,split='train',resize=(256,256)):

        
        self.annotations = json.load(open(os.path.join(data_path, 
                                                       'ridge_points', f"{split}.json")))
        self.resize=resize

        if split=="train" or split== "augument":
            self.heatmap_transform=transforms.Compose([
                    TensorhortonFlip(),
                    TensorVerticalFlip(),
                    TensorRotate(),
                    TensorNorm()
                ])
        elif split=='val' or split=='test':
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
        {
            'image_name':os.path.basename(img_path),
            'image_path':img_path,
            'point_number':self.point_number,
            'heatmap_path':save_path,
            'ridge':[{
                'coordinate':preds_list[i],
                'value':maxvals_list[i]
            } for i in range(len(maxvals_list))]
        }
        '''
        # Load the image and label
        annotation = self.annotations[idx]
        heatmap=torch.load(annotation['heatmap_path'])
        label=annotation['class']

        # Transforms the image
        heatmap=F.interpolate(heatmap[None,None,:,:]
                              ,size=self.resize,
                                mode='bilinear',
                                  align_corners=False).squeeze()
        heatmap=self.heatmap_transform(heatmap)
        # Store esscencial data for visualization (Gram)
        meta={}
        meta['image_path']=annotation['image_path']

        return heatmap,label,meta
    
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