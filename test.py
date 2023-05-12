import json
import os
import torch
from cleansing import crop_square
from torchvision import transforms
from config import get_config
from utils_ import get_instance,ContrastEnhancement
import models
from sklearn.metrics import accuracy_score
from ridgeLocateModule import RidgeLocateProcesser
import numpy as np
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs.MODEL.NAME,args.configs,
                         num_classes=args.configs.NUM_CLASS)
criterion=torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(args.save_name))
print(f"load the checkpoint in {args.save_name}")
model.eval()

# Create the dataset and data loader
data_path=os.path.join(args.path_tar)
test_data_list=json.load(open(os.path.join(data_path, 'crop_ridge_annotations_baseline', "test.json")))


# Create the visualizations directory if it doesn't exist
crop_per_image=4
crop_processer=RidgeLocateProcesser(crop_per_image,20)
all_targets = []
all_outputs = []
test_transforms=transforms.Compose([
                ContrastEnhancement(),
                transforms.Resize(args.configs.IMAGE_RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
            ])
with torch.no_grad():
    for test_data in test_data_list:
        image_path=test_data['image_path']
        label=test_data['class']
        preds,maxvals=crop_processer(image_path)
        cnt=0
        predict_labels_image=0
        for x,y in preds:
            crop_image=crop_square(image_path,x=x,y=y,
                        width=args.crop_width)
            img=test_transforms(crop_image).to(device)
            
            outputs = model(img.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1).squeeze().cpu()
            predict_labels_image=max(predict_labels_image,int(predicted_labels))

        all_targets.append(label)
        all_outputs.extend(predict_labels_image)
acc = accuracy_score(np.array(all_targets),np.array(all_outputs))
print(f"Finished testing! Test acc {acc:.4f}")
