import json
import os
import torch
from torchvision import transforms
from config import get_config
from utils_ import to_device, crop_square, ContrastEnhancement
from models import build_model
import numpy as np
from PIL import Image
from utils_ import acc, auc, auc_sens

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model, _ = build_model(configs=args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, f"{args.split_name}_{args.save_name}")))
print("load the checkpoint in {}".format(os.path.join(
    args.save_dir, f"{args.split_name}_{args.save_name}")))
model.eval()

# Create the dataset and data loader

# Create the visualizations directory if it doesn't exist
with open(os.path.join(args.data_path, 'split', f'{args.split_name}.json'), 'r') as f:
    split_list = json.load(f)['test']
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)

all_targets = []
all_outputs = []
all_scores = []
print(f"test name number {len(split_list)}")
heatmap_process = transforms.Compose([
    transforms.Resize(args.configs['image_resize']),
    transforms.ToTensor()
])
label_map = {1: 0, 2: 1,3:2}
os.makedirs(os.path.join(args.result_path,
            'stage123_crop_image'), exist_ok=True)
img_process = transforms.Compose([
    ContrastEnhancement(),
    transforms.Resize(args.configs["image_resize"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])])
vessel_process = transforms.Compose([
    transforms.Resize(args.configs["vessel_resize"]),
    transforms.ToTensor()])
def decide_statgy(predict1,predict2):
    max_1=max(predicted_label1)
    max_2=max(predict2)
    if max_2==3:
        return 2
    else:
        return max_1
with torch.no_grad():
    for image_name in split_list:
        data = data_dict[image_name]
        if data['stage'] not in [1,2]:
            continue
        label = label_map[data['stage']]
        crop_img_list = []
        crop_vessel_list = []
        cnt = 0
        for x, y in data['ridge_seg']['coordinate']:
            croped_img = crop_square(data['image_path'], x, y, args.configs['crop_width'],
                                     save_path=os.path.join(args.result_path, 'stage123_crop_image', f"{data['id']}_{str(cnt)}.jpg")).convert('RGB')
            croped_vessel = crop_square(data['vessel_path'], x, y, args.configs['crop_width'],
                                     save_path=os.path.join(args.result_path, 'stage123_crop_image', f"{data['id']}_{str(cnt)}_vessel.jpg")).convert('RGB')
            cnt += 1
            crop_img_list.append(img_process(croped_img).unsqueeze(0))
            crop_vessel_list.append(vessel_process(croped_vessel).unsqueeze(0))

        inputs=to_device(
            x=(torch.cat(crop_img_list,dim=0),torch.cat(crop_vessel_list,dim=0)),
            device=device
        )
        out1,out2 = model(inputs)
        probs1 = torch.softmax(out1, dim=1)
        predicted_label1 = torch.argmax(out1, dim=1).squeeze().cpu()
        probs2 = torch.softmax(out2, dim=1)
        predicted_label2 = torch.argmax(out2, dim=1).squeeze().cpu()+1
        predicted_label=decide_statgy(predicted_label1,predicted_label2)
        all_targets.append(label)
        all_outputs.append(int(predicted_label))
all_targets = np.array(all_targets)
all_outputs = np.array(all_outputs)
print(f"acc: {acc(all_targets,all_outputs)} | auc: {auc_sens(all_targets,all_outputs)}")
print("Finished testing!")
