import json
import os
import torch
from cleansing import crop_square
from torchvision import transforms
from config import get_config
from utils_ import get_instance,ContrastEnhancement,sensitive_score,TensorNorm
import models
from sklearn.metrics import accuracy_score
from ridgeLocateModule import RidgeLocateProcesser
import numpy as np
import cv2
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs.MODEL.NAME,args.configs,
                         num_classes=args.configs.NUM_CLASS,mode=args.model_mode)
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
all_targets = []
all_outputs = []
heatmap_transforms=TensorNorm()
print({f"teat name number {len(test_data_list)}"})
with torch.no_grad():
    for test_data in test_data_list:
        image_path=test_data['image_path']
        label=test_data['class']
        image_name= os.path.basename(image_path)
        heatmap=torch.load(os.path.join(data_path,'ridge_heatmap',image_name.split('.')[0]+'.pt'))
        heatmap=heatmap_transforms(heatmap).unsqueeze(0).to(device)
        
        outputs = model(heatmap)
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1).squeeze().cpu()
        all_targets.append(label)
        all_outputs.append(predicted_labels)
acc = accuracy_score(np.array(all_targets),np.array(all_outputs))
sens=sensitive_score(all_targets,all_outputs,test_data_list)
print(f"Finished testing! Test acc {acc:.4f} sensitive: {(sens):.4f}")
