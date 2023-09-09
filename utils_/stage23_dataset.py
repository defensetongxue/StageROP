import torch.utils.data as data
from PIL import Image,ImageEnhance  
import os
import os.path
import torch
from torchvision import transforms
import json
class stage23_Dataset(data.Dataset):
    def __init__(self, data_path,split='train',split_name='0',
                 vessel_resize=(256,256)):
        # input the vessel path
        self.split=split
        with open(os.path.join(data_path,'stage_rop','crop_annotations.json'),'r') as f:
            self.annotation=json.load(f)
        with open(os.path.join(data_path,'stage_rop','crop_split',f"{split_name}.json"),'r') as f:
            split_all = json.load(f)[split]
        self.split_list=[]
        for crop_name in split_all:
            if self.annotation[crop_name]['Stage'] in [2,3]:
                self.split_list.append(crop_name)

        self.vessel_resize=transforms.Resize(vessel_resize)
        self.vessel_enhance=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                ])
        self.vessel_transform=transforms.ToTensor()
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        '''
        "crop_from": image_name,
                    "stage":data["stage"],
                    "crop_vessel_path":vessel_crop_path,
                    "crop_image_path":image_crop_path
        '''
        # Load the image and label
        crop_name = self.split_list[idx]
        data=self.annotation[crop_name]
        label=data['stage']
        vessel_path=data["crop_vessel_path"]
        vessel=Image.open(vessel_path)
        vessel=self.vessel_resize(vessel)
        if self.split == "train" :
            vessel=self.vessel_enhance(vessel)
        vessel=self.vessel_transform(vessel).repeat(3,1,1)
        meta={}
        meta['crop_name']=crop_name
        return vessel,label,meta
    

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
    
