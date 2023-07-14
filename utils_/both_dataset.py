import torch.utils.data as data
from PIL import Image,ImageEnhance  
import os
import os.path
import torch
from torchvision import transforms
import json
class both_Dataset(data.Dataset):
    
    def __init__(self, data_path,split='train',
                 img_resize=(256,256),
                 heatmap_resize=(256,256),
                 vessel_resize=(256,256)):

        
        self.annotations = json.load(
            open(os.path.join(data_path,'ridge_crop','annotations',f"{split}.json")))
        self.split=split
        self.img_resize=transforms.Compose(
            [ContrastEnhancement(),transforms.Resize(img_resize)])
        self.vessel_resize=transforms.Resize(vessel_resize)
        self.heatmap_resize=transforms.Resize(heatmap_resize)
        self.img_enhance=transforms.Compose([
                ContrastEnhancement(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                
                ])
        self.img_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])
            ])
        self.vessel_transform=transforms.ToTensor()
        self.heatmap_transform=transforms.ToTensor()
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        '''
        "image_name":data['image_name'],
        "image_path":data["image_path"],
        "class":data["class"],
        "crop_name":crop_name,
        "crop_vessel_path":vessel_crop_path,
        "crop_image_path":image_crop_path,
        "heatmap_path":heatmap_path,
    '''
        # Load the image and label
        annotation = self.annotations[idx]
        
        img=Image.open(annotation["crop_image_path"])
        img=self.img_resize(img)
        vessel=Image.open(annotation["crop_vessel_path"])
        vessel=self.vessel_resize(vessel)
        heatmap=Image.open(annotation["heatmap_path"])
        label=annotation['class']

        if self.split == "train" :
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.img_enhance(img)
            torch.manual_seed(seed)
            vessel = self.img_enhance(vessel)
            torch.manual_seed(seed)
            heatmap=self.img_enhance(heatmap)
            
        # Transforms the image
        img=self.img_transform(img)
        vessel=self.vessel_transform(vessel).repeat(3,1,1)
        heatmap=self.heatmap_transform(heatmap).unsqueeze(0).repeat(3,1,1)
        # Store esscencial data for visualization (Gram)
        meta={}
        meta['image_path']=annotation['crop_name']

        return [img,vessel,heatmap],label,meta
    
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