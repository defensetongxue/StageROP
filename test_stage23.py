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
    transforms.Resize(args.configs['vessel_resize']),
    transforms.ToTensor()
])
label_map = {2: 0, 3: 1}
os.makedirs(os.path.join(args.result_path,
            'stage23_crop_image'), exist_ok=True)
vessel_process = transforms.Compose([
    transforms.Resize(args.configs["vessel_resize"]),
    transforms.ToTensor()])
def decide_statgy(predict):
    return max(predict)
with torch.no_grad():
    for image_name in split_list:
        data = data_dict[image_name]
        if data['stage'] not in [2,3]:
            continue
        label = label_map[data['stage']]
        vessel = Image.open(data['vessel_path']).convert('RGB')
        crop_vessel_list = []
        cnt = 0
        for x, y in data['ridge_seg']['coordinate']:
            croped_vessel = crop_square(data['image_path'], x, y, args.configs['crop_width'],
                                     save_path=os.path.join(args.result_path, 'stage23_crop_image', f"{data['id']}_{str(cnt)}_vessel.jpg")).convert('RGB')
            cnt += 1
            crop_vessel_list.append(vessel_process(croped_vessel).unsqueeze(0))

        inputs=to_device(
            x=torch.cat(crop_vessel_list,dim=0),
            device=device
        )
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1).squeeze().cpu()
        predicted_label=decide_statgy(predicted_labels)
        all_targets.append(label)
        all_outputs.append(int(predicted_label))
        all_scores.append(probs.detach().cpu())
all_targets = np.array(all_targets)
all_outputs = np.array(all_outputs)
all_scores = torch.cat(all_scores, dim=0).numpy()
print(f"acc: {acc(all_targets,all_outputs)} | auc: {auc_sens(all_targets,all_outputs)}")
print("Finished testing!")
