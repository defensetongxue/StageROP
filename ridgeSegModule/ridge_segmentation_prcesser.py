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

# class ridge_segmentation_processer():
#     def __init__(self,mode,point_number,point_dis=50):
#         self.mode=mode
#         assert mode in ['mask','point']
#         if mode=='point':
#             self.point_number=point_number
#             self.point_dis=point_dis

#         self.model=FR_UNet().cuda()
#         self.model.load_state_dict(
#                 torch.load('./ridgeSegModule/checkpoint/ridge_seg.bth'))
#         self.img_transforms=transforms.Compose([
#             transforms.Resize((600,800)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.4623, 0.3856, 0.2822],
#                 std=[0.2527, 0.1889, 0.1334])])

#     def __call__(self,img_path,mask_save_path=None):
#         img=Image.open(img_path)
#         img=self.img_transforms(img).unsqueeze(0) # build batch as 1
#         mask=self.model(img.cuda()).cpu()
#         mask=torch.sigmoid(mask).numpy()
#         mask=zoom(mask,2)
#         if mask_save_path:
#             torch.save(mask,mask_save_path)
#         if self.mode=='point':
#             maxvals_list, preds_list=k_max_values_and_indices(
#                 mask,self.point_number,self.point_dis)
#             maxvals_list=maxvals_list.tolist()
#             preds_list=preds_list.tolist()
#             data={
#                 'image_name':os.path.basename(img_path),
#                 'image_path':img_path,
#                 'point_number':self.point_number,
#                 'ridge':[{
#                     'coordinate':preds_list[i],
#                     'value':maxvals_list[i]
#                 } for i in len(maxvals_list)]
#             }
#             return mask,data
#         return mask
def decompose_image_into_tensors(image):
    # Assumes the input is a PyTorch tensor
    height, width = image.shape[1], image.shape[2]
    # Split the image tensor into two along the height
    first_half, second_half = torch.split(image, height//2, dim=1)
    # Then split each half into two along the width
    first_half_tensors = torch.split(first_half, width//2, dim=2)
    second_half_tensors = torch.split(second_half, width//2, dim=2)
    # Return a list of all four image parts
    return list(first_half_tensors) + list(second_half_tensors)

def compose_tensors_into_image(tensors_list):
    # Assumes the input is a list of four tensors
    first_half = torch.cat(tensors_list[:2], dim=2)  # Concatenate along width
    second_half = torch.cat(tensors_list[2:], dim=2)  # Concatenate along width
    return torch.cat([first_half, second_half], dim=1)  # Concatenate along height

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
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])

    def __call__(self,img_path,mask_save_path=None):
        img=Image.open(img_path)
        decomposed_images = decompose_image_into_tensors(self.img_transforms(img))
        composed_mask = []
        for decomposed_image in decomposed_images:
            decomposed_image = decomposed_image.unsqueeze(0).cuda()  # make sure it's a batch of 1
            mask_part = self.model(decomposed_image).cpu()
            mask_part = torch.sigmoid(mask_part)
            composed_mask.append(mask_part)

        mask = compose_tensors_into_image(composed_mask)
        
        if mask_save_path:
            torch.save(mask,mask_save_path)

        if self.mode=='point':
            mask_numpy = mask.numpy()
            maxvals_list, preds_list=k_max_values_and_indices(
                mask_numpy,self.point_number,self.point_dis)
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
            return mask, data
        return mask
