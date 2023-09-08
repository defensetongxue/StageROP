import torch.utils.data as data
from PIL import Image,ImageEnhance  
import torch.nn.functional as F
import os
import os.path
import torch
from torchvision import transforms
import json
class sick_Dataset(data.Dataset):
    '''
        └───data
            │
            └───'heatmap'
            │   │
            │   └───001.jpg
            │   └───002.jpg
            │   └───...
            │
    '''
    def __init__(self, data_path,split='train',heatmap_resize=(256,256),split_name='0'):

        with open(os.path.join(os.path.join(data_path,'annotations', f"{split_name}.json"),'r')) as f:
            self.annotation=json.load(f)
        with open(os.path.join(data_path,'split',f"{split_name}.json"),'r') as f:
            self.split_list=json.load(f)[split]
        self.split=split
        self.heatmap_resize=transforms.Resize((heatmap_resize))
        self.heatmap_enhance=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                ])
        self.heatmap_transform=transforms.ToTensor()
    def __len__(self):
        return len(self.split_list)
    
    def __getitem__(self, idx):
        '''
        '''
        # Load the image and label
        image_name = self.split_list[idx]
        data=self.annotation[image_name]
        heatmap=Image.open(data['ridge_seg']['heatmap_path'])
        label=data['stage']
        if label>0:
            label=1
        heatmap=self.heatmap_resize(heatmap)
        if self.split =='train':
            heatmap=self.heatmap_enhance(heatmap)
        heatmap=self.heatmap_transform(heatmap).repeat(3,1,1)
        meta={}
        meta['image_path']=data['image_path']

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
    