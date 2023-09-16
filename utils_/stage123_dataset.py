import torch.utils.data as data
from PIL import Image,ImageEnhance  
import os
import os.path
import torch
from torchvision import transforms
import json
class stage123_Dataset(data.Dataset):
    def __init__(self, data_path,configs,split='train',split_name='0'):
        # input the vessel path
        self.split=split
        with open(os.path.join(data_path,'stage_rop','crop_annotations.json'),'r') as f:
            self.annotation=json.load(f)
        with open(os.path.join(data_path,'stage_rop','crop_split',f"{split_name}.json"),'r') as f:
            self.split_list = json.load(f)[split]

        self.img_resize=transforms.Compose([ContrastEnhancement(),
                                            transforms.Resize(configs['image_resize'])])
        
        self.img_enhance=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                ])
        self.img_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])])
        
        self.vessel_resize=transforms.Resize(configs['vessel_resize'])
        self.vessel_enhance=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                ])
        self.vessel_transform=transforms.ToTensor()
        self.label_map={
            1:[0,0],2:[1,0],3:[1,1]
        }
    def __len__(self):
        return len(self.split_list)
    
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
        label=self.label_map[label]
        image_path=data["crop_image_path"]
        image=Image.open(image_path).convert('RGB')
        image=self.img_resize(image)
        vessel=Image.open(data["crop_vessel_path"]).convert('RGB')
        vessel=self.vessel_resize(vessel)
        if self.split == "train" :
            image=self.img_enhance(image)
            vessel=self.vessel_enhance(vessel)

        image=self.img_transform(image)
        vessel=self.vessel_transform(vessel)
        meta=crop_name
        return [image,vessel],label,meta
    

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