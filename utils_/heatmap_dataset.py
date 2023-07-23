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
                                                       'ridge_crop','heatmap_annotations', f"{split}.json")))
        self.resize=resize

        self.split=split
        self.heatmap_resize=transforms.Resize((resize))
        self.heatmap_enhance=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                ])
        self.heatmap_transform=transforms.ToTensor()
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
        # heatmap=Image.open(annotation['image_path'])
        heatmap=Image.open(annotation['heatmap_path'])
        label=annotation['class']
        if label>0:
            label=1
        heatmap=self.heatmap_resize(heatmap)
        if self.split =='train':
            heatmap=self.heatmap_enhance(heatmap)
        heatmap=self.heatmap_transform(heatmap).repeat(3,1,1)
        # heatmap=self.heatmap_transform(heatmap)
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
    