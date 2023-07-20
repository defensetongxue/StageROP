import torch.utils.data as data
from PIL import Image,ImageEnhance  
import torch.nn.functional as F
import os
import os.path
import torch
from torchvision import transforms
import json
class val_Dataset(data.Dataset):
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
    def __init__(self, data_path,split='train'):

        
        self.annotations = json.load(open(os.path.join(data_path, 
                                                       'ridge_crop','val_annotations', f"{split}.json")))
        self.idx_list=list(self.annotations.keys())
    def __len__(self):
        return len(self.idx_list)
    
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
        image_select = self.idx_list[idx]
        data=self.annotations[image_select]['value']
        data=torch.tensor(data)
        label=self.annotations[image_select]['class']
        # Store esscencial data for visualization (Gram)
        meta={'image_name':image_select}
        return data,label,meta