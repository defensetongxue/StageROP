import json
import os
import torch
from torchvision import transforms
from config import get_config
from utils_ import get_instance
import models.heatmap as models
import numpy as np
from PIL import Image
from utils_ import acc,auc,auc_sens

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs.MODEL.NAME,args.configs,
                         num_classes=args.configs.NUM_CLASS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(args.save_name))
print(f"load the checkpoint in {args.save_name}")
model.eval()

# Create the dataset and data loader
data_path=os.path.join(args.path_tar)

# Create the visualizations directory if it doesn't exist
with open(os.path.join(data_path,'ridge_points','test.json'),'r') as f:
    ridge_seg_list=json.load(f)
all_targets = []
all_outputs = []
all_scores=[]
heatmap_transform=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()])
print(f"test name number {len(ridge_seg_list)}")
cnt_max=5
with torch.no_grad():
    for data in ridge_seg_list:
        image_path=data['image_path']
        label=data['class']
        image_name= data['image_name']
        heatmap=Image.open(data["heatmap_path"])
        heatmap=heatmap_transform(heatmap).repeat(3,1,1).to(device)
        outputs = model(heatmap.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1).squeeze().cpu()
        all_targets.append(label)
        all_outputs.append(int(predicted_labels))
        all_scores.append(probs.detach().cpu())
all_targets=np.array(all_targets)
all_outputs=np.array(all_outputs)
all_scores=torch.cat(all_scores,dim=0).numpy()
print("Finished testing!")
print(f"acc: {acc(all_targets,all_outputs)} | auc: {auc(all_targets,all_scores)}")
all_targets[all_targets>0]=1
all_outputs[all_outputs>0]=1
print(f"sens acc: {acc(all_targets,all_outputs)}| {auc_sens(all_targets,all_outputs)}" )
