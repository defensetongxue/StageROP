import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from scipy.ndimage import zoom
from .models import FR_UNet

def k_max_values_and_indices(scores, k,r=50):

    preds_list = []
    maxvals_list = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)

        maxval = scores[idx]

        maxvals_list.append(maxval)
        preds_list.append(idx)

        # Clear the square region around the point
        x, y = idx[0], idx[1]
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.shape[0], x + r // 2), min(scores.shape[1], y + r // 2)
        # print(scores.shape,xmin,xmax,ymin,ymax)
        scores[ xmin:xmax,ymin:ymax] = -9
        # print(scores[ xmin:xmax,ymin:ymax])
        # raise
    maxvals_list=np.array(maxvals_list,dtype=np.float32)
    preds_list=np.array(preds_list,dtype=np.float32)
    return maxvals_list, preds_list

class ridge_segmentation_processer():
    def __init__(self,mode,point_number,point_dis=50):
        self.mode=mode
        assert mode in ['mask','point']
        if mode=='point':
            self.point_number=point_number
            self.point_dis=point_dis

        self.model=FR_UNet().cuda()
        self.model.load_state_dict(
                torch.load('./ridgeSegModule/checkpoint/ridge_seg.bth'))
        self.img_transforms=transforms.Compose([
            transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
    def __call__(self,img_path,mask_save_path=None):
        img=Image.open(img_path)
        img=self.img_transforms(img).unsqueeze(0) # build batch as 1
        mask=self.model(img.cuda()).cpu()
        mask=torch.sigmoid(mask).numpy()
        mask=zoom(mask,2)
        if mask_save_path:
            torch.save(mask,mask_save_path)
        if self.mode=='point':
            maxvals_list, preds_list=k_max_values_and_indices(
                mask,self.point_number,self.point_dis)
            maxvals_list=maxvals_list.tolist()
            preds_list=preds_list.tolist()
            data={
                'image_name':os.path.basename(img_path),
                'image_path':img_path,
                'point_number':self.point_number,
                'ridge':[{
                    'coordinate':preds_list[i],
                    'value':maxvals_list[i]
                } for i in len(maxvals_list)]
            }
            return mask,data
        return mask