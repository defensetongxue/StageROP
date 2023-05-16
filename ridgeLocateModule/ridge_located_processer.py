from .models import hrnet
import torch
from PIL import ImageEnhance,Image
import torchvision.transforms as transforms
from .model_config import get_model_config
import math
import json
import numpy as np
class RidgeLocateProcesser():
    def __init__(self,select_number=3,region_width=4):
        config=get_model_config()
        self.model,_ = hrnet(config)
        checkpoint = torch.load(
            './ridgeLocateModule/checkpoint/best.pth')
        self.model.load_state_dict(checkpoint)
        self.model.cuda()

        self.transforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize(config.IMAGE_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4623, 0.3856, 0.2822],
                 std=[0.2527, 0.1889, 0.1334])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
        ])
        self.select_number=select_number
        self.region_width=region_width
        
        self.w_r=config.IMAGE_SIZE[0]/config.IMAGE_RESIZE[0]
        self.h_r=config.IMAGE_SIZE[1]/config.IMAGE_RESIZE[1]

    def __call__(self, img_path,save_path=None):
        img = Image.open(img_path).convert('RGB')
        img=self.transforms(img).unsqueeze(0)
        
        output=self.model(img.cuda())
        score_map = output.data.cpu()
        preds,maxvals = decode_preds(score_map,self.select_number,self.region_width)
        # Ensure preds and maxvals are NumPy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.squeeze(0).numpy()
        if isinstance(maxvals, torch.Tensor):
            maxvals = maxvals.squeeze().numpy()
        # Draw landmarks on the image
        preds=np.array(preds)
        if len(preds.shape)<=1:
            preds=preds.reshape(1,-1)
        preds[:,0]*=self.w_r
        preds[:,1]*=self.h_r
        
        if save_path:
            with open(save_path,'w') as f:
                json.dump({
                "coordinate":preds,
                "maxvals":maxvals
            },f)
        return preds,maxvals
    def generate_heatmap(self,img_path,save_path=None):
        img = Image.open(img_path).convert('RGB')
        img=self.transforms(img).unsqueeze(0)
        
        output=self.model(img.cuda())
        score_map = output.data.squeeze().cpu()
        if save_path:
            torch.save(score_map,save_path)
            
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img
    

def decode_preds(output, visual_num=3,region_width=20):

    assert output.dim() == 4, 'Score maps should be 4-dim'
    assert output.shape[0]==1 ,'visual should be batch==1'
    output=output.squeeze() # (width, height)
    map_width, map_height = output.shape[-2], output.shape[-1]
    coords,maxval = get_preds(output, visual_num,region_width)  # float type

    # pose-processing
    for k in range(coords.shape[0]):
        hm = output.clone()
        px = int(math.floor(coords[k][0]))
        py = int(math.floor(coords[k][1]))
        if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
            coords[k] += diff.sign() * 0.25
    preds = coords.clone()

    # Transform back
    return preds * 4,maxval  # heatmap is 1/4 of the original image


def get_preds(scores, number, r=20):
    """
    input scores is 2d-tensor heatmap
    number is the number of select point

    return shape should be pres=(number,2) maxvals=(number)
    """

    preds_list = []
    maxvals_list = []

    temp_scores = scores.clone()

    for _ in range(number):
        maxval, idx = torch.max(temp_scores.view(-1), dim=0)
        maxval = maxval.flatten()
        idx = idx.view(-1, 1) + 1

        pred = idx.repeat(1, 2).float()
        pred[:, 0] = (pred[:, 0] - 1) % scores.size(1) + 1
        pred[:, 1] = torch.floor((pred[:, 1] - 1) / scores.size(1)) + 1

        maxvals_list.append(maxval.item())
        preds_list.append(pred.squeeze())
        # Clear the square region around the point
        x, y = int(pred[0, 0].item()), int(pred[0, 1].item())
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.size(1), x + r // 2), min(scores.size(0), y + r // 2)
        temp_scores[ymin:ymax, xmin:xmax] = 0

    preds = torch.stack(preds_list)
    maxvals = torch.tensor(maxvals_list)
    return preds, maxvals
