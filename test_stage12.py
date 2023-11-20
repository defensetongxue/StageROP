import json
import os
import torch
from torchvision import transforms
from config import get_config
from utils_ import  crop_square
from models import build_model
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import numpy as np
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
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)
with open(os.path.join(args.data_path,'split',f"{args.split_name}.json"),'r') as f:
    split_dict=json.load(f)['test']
print(f"test name number {len(split_dict)}")
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
probs_image=[]
visual_dir='./experiments/'+args.configs['model']['name']
print(visual_dir)
os.makedirs(visual_dir,exist_ok=True)
os.system(f"rm -rf {visual_dir}/*")
def draw_square(image_path, points, points3, width,save_path, values=None, values3=None, font_size=40, value_font_size=24,extra_text="no"):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    # Load the font for texts and values
    font_path = './arial.ttf'
    text_font = ImageFont.truetype(font_path, font_size)
    value_font = ImageFont.truetype(font_path, value_font_size)

    # Function to draw boxes with specified color and values
    def draw_boxes(points, values, box_color, value_color):
        for i, (x, y) in enumerate(points):
            # Define the position of the squares
            top_left = (x - width // 2, y - width // 2)
            bottom_right = (x + width // 2, y + width // 2)
            # Draw the squares
            draw.rectangle([top_left, bottom_right], outline=box_color, width=5)
            
            # Draw the values on the top left of the square, if provided
            if values is not None:
                value = round(values[i], 2)
                val_text = str(value)
                # Calculate position for the text
                val_x, val_y = top_left
                draw.text((val_x, val_y - value_font_size), val_text, fill=value_color, font=value_font)

    # Draw green boxes for the first set of points and values
    if points:
        # Sample indices if there are more than 3 points
        if len(points) > 3:
            sampled_indices = random.sample(range(len(points)), 3)
            points = [points[i] for i in sampled_indices]
            values = [values[i] for i in sampled_indices] if values else None
        draw_boxes(points, values, "green", "yellow")

    # Draw red boxes for the second set of points and values
    if points3:
        # Sample indices if there are more than 3 points
        if len(points3) > 3:
            sampled_indices = random.sample(range(len(points3)), 3)
            points3 = [points3[i] for i in sampled_indices]
            values3 = [values3[i] for i in sampled_indices] if values3 else None
        draw_boxes(points3, values3, "red", "yellow")

    # Draw the texts with the new font
    # draw.text((10, 10), left_text, fill="white", font=text_font)
    # draw.text((10, 250), extra_text, fill="white", font=text_font)
    # text_bbox = draw.textbbox((0, 0), right_text, font=text_font)
    # text_width = text_bbox[2] - text_bbox[0]
    # draw.text((img.width - text_width - 10, 10), right_text, fill="white", font=text_font)

    # Save the image
    img.save(save_path)


with torch.no_grad():
    for image_name in split_dict:
        data = data_dict[image_name]
        label = data['stage']
        # if label==0 or label>3:
        #     continue
        if 'ridge' not in data:
            # print(image_name)
            continue
        label-=1
        crop_img_list = []
        
        # Process each coordinate to get the cropped image and its prediction
        candidate_list=data['ridge']["ridge_coordinate"]
        if len(candidate_list)>1:
            candidate_list=random.sample(candidate_list,1)
        for cnt, (x, y) in enumerate(candidate_list):
            crop_img = crop_square(data['image_path'], x, y, args.configs['crop_width']).convert('RGB')
            processed_img = img_process(crop_img).unsqueeze(0)
            crop_img_list.append(processed_img)
        crop_img_list=torch.cat(crop_img_list,dim=0).to(device)
        output = model(crop_img_list).cpu()
        probs = torch.softmax(output, dim=1)
        predict=torch.argmax(probs,dim=1)
        predicted_label = torch.max(predict)  
        prob_list=[]
        for i,preds_patch in enumerate(predict):
            if preds_patch==predicted_label:
                prob_list.append(probs[i].unsqueeze(0))
        # print(prob_list)
        prob_list=torch.cat(prob_list,dim=0)
        prob_list=prob_list.mean(dim=0).unsqueeze(0)
        
        probs_image.append(prob_list)
        #########################################################
        # vals=probs[:,1].squeeze().tolist()
        # vals3=probs[:,2].squeeze().tolist()
        # selected_points = [coord for idx, coord in enumerate(candidate_list) if predict[idx] == 1]
        # selected_values = [val for idx, val in enumerate(vals) if predict[idx] == 1]
        # selected_points3 = [coord for idx, coord in enumerate(candidate_list) if predict[idx] == 2]
        # selected_values3 = [val for idx, val in enumerate(vals3) if predict[idx] == 2]
        # assert len(selected_points3)==len(selected_values3),(len(candidate_list),len(vals))
        # assert len(selected_points)==len(selected_values),(probs,predict,vals3)
        
        # # print(len(selected_points3),len(selected_values3))
        # # Call draw_square to visualize and save the image
        # if data['ridge']["vessel_abnormal_number"]>0:
        #     extra_text="Yes"
        # else:
        #     extra_text="No"
        # draw_square(
        #     image_path=data['image_path'],
        #     points=selected_points,
        #     points3=selected_points3 ,
        #     width=args.configs['crop_width'],
        #     # left_text=f"ZhangYuan: {split_dict[image_name]['zy']}",
        #     # right_text=f"XieSJ: {split_dict[image_name]['xie']}",
        #     save_path=os.path.join(visual_dir,image_name),  # Specify your save path format
        #     values=selected_values,
        #     values3=selected_values3,
        #     extra_text=extra_text,
        # )
        #########################################################
         
        # Aggregating results for the entire ima
        image_labels.append(int(label))
        image_predictions.append(int(predicted_label))  # Average probabilities
image_labels=np.array(image_labels)
image_predictions=np.array(image_predictions)
probs_image=torch.cat(probs_image,dim=0).numpy()

# Convert lists to arrays for metric calculations
acc=accuracy_score(image_labels,image_predictions)
auc=roc_auc_score(image_labels,probs_image, multi_class='ovr')
acc=round(acc,4)
auc=round(auc,4)
with open('result.json','r') as f:
  res=json.load(f)
record_name=f"{str(args.configs['crop_width'])}_{str(args.configs['vessel_disctance_threshold'])}"
res[record_name]={"acc":acc,"auc":auc}
with open("result.json",'w') as f:
  json.dump(res,f)