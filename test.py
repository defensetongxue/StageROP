import json
import os
import torch
from torchvision import transforms
from config import get_config
from utils_ import to_device, crop_square
from models import build_model
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
import random
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")
save_name=args.configs['model']['name']+'.pth'
# Create the model and criterion
model, _ = build_model(configs=args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, f"{args.split_name}_{save_name}")))
print("load the checkpoint in {}".format(os.path.join(
    args.save_dir, f"{args.split_name}_{save_name}")))
model.eval()

# Create the dataset and data loader

# Create the visualizations directory if it doesn't exist
with open(os.path.join(args.data_path, 'split',f"{args.split_name}.json"), 'r') as f:
    split_list = json.load(f)['test']
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)

print(f"test name number {len(split_list)}")
os.makedirs(os.path.join(args.result_path,args.split_name), exist_ok=True)
img_process = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])])


# Function to decide the label based on the maximum probability
def decide_strategy(predicted_probs):
    # Use max probability to determine label
    return np.argmax(np.max(predicted_probs, axis=0))
# Initialize lists to hold the aggregated results
image_labels = []
image_predictions = []

save_dir=os.path.join(args.data_path,'stage_bbox')
os.makedirs(save_dir,exist_ok=True)
os.system("rm -rf {visual_dir}/*")
with torch.no_grad():
    for image_name in split_list:
        data = data_dict[image_name]
        label = data['stage']
        if label==0 or label>3:
            continue
        if 'ridge' not in data:
            print(image_name)
            continue
        label-=1
        crop_img_list = []
        
        # Process each coordinate to get the cropped image and its prediction
        candidate_list=data['ridge_seg']["point_list"]
        if len(candidate_list)==0:
            continue
        if len(candidate_list)>=12:
            candidate_list=random.sample(candidate_list,12)
        for cnt, (x, y) in enumerate(candidate_list):
            crop_img = crop_square(data['enhanced_path'], x, y, args.configs['crop_width']).convert('RGB')
            processed_img = img_process(crop_img).unsqueeze(0)
            crop_img_list.append(processed_img)
        crop_img_list=torch.cat(crop_img_list,dim=0).to(device)
        output = model(crop_img_list).cpu()
        probs = torch.softmax(output, dim=1)
        predict=torch.argmax(probs,dim=1)
        predicted_label = torch.max(predict)  
        vals=probs[:,1].squeeze().tolist()
        selected_points = [coord for idx, coord in enumerate(candidate_list) if predict[idx] == 1]
        selected_values = [val for idx, val in enumerate(vals) if predict[idx] == 1]
        # Call draw_square to visualize and save the images
        save_name=os.path.join(save_dir,image_name)
        save_value=[]
        for value in selected_values:
            value=round(float(value),2)
            save_value.append(value)
        save_point=[]
        for x,y in selected_points:
            save_point.append([int(x),int(y)])
        data_dict[image_name]['stage_result']={
            "points":save_point,
            "values":save_value,
            "box_width":args.configs['crop_width']}
        # Aggregating results for the entire ima
        image_labels.append(label)
        image_predictions.append(predicted_label)  # Average probabilities
# Convert lists to arrays for metric calculations
image_labels = np.array(image_labels)
image_predictions = np.array(image_predictions)
# acc = accuracy_score(image_labels, image_predictions)
# auc = roc_auc_score(image_labels, image_predictions) 
# auc = roc_auc_score(image_labels, 0) 
# print(f"Acc: {acc}, Auc: {auc}")

# with open(os.path.join(args.data_path, 'annotations.json'), 'w') as f:
#     json.dump(data_dict,f)
with open('./new.json', 'w') as f:
    json.dump(data_dict,f)