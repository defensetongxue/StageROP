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
with open('./confusion.json', 'r') as f:
    split_list = json.load(f)
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

visual_dir='./experiments/'+args.configs['model']['name']
os.makedirs(visual_dir,exist_ok=True)
os.system("rm -rf {visual_dir}/*")
def draw_square(image_path, points, width, left_text, right_text, save_path, values=None, font_size=40, value_font_size=24):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    # Load the font for texts and values
    font_path = './arial.ttf'
    text_font = ImageFont.truetype(font_path, font_size)
    value_font = ImageFont.truetype(font_path, value_font_size)

    # Draw the squares if there are points
    if points:
        if len(points) > 3:
            points = random.sample(points, 3)
            if values is not None:
                values = random.sample(values, 3)

        for i, (x, y) in enumerate(points):
            # Define the position of the squares
            top_left = (x - width // 2, y - width // 2)
            bottom_right = (x + width // 2, y + width // 2)
            # Draw the squares
            draw.rectangle([top_left, bottom_right], outline="green", width=5)
            
            # Draw the values on the top left of the square
            if values is not None:
                # print(values[i])
                value=round(values[i], 2)
                # print(value)
                val_text = str(value)
                # Calculate position for the text
                val_x, val_y = top_left
                draw.text((val_x, val_y - value_font_size), val_text, fill="yellow", font=value_font)

    # Draw the texts with the new font
    draw.text((10, 10), left_text, fill="white", font=text_font)
    text_bbox = draw.textbbox((0, 0), right_text, font=text_font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text((img.width - text_width - 10, 10), right_text, fill="white", font=text_font)

    # Save the image
    img.save(save_path)


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
        candidate_list=data['ridge']["ridge_coordinate"]
        if len(candidate_list)>=12:
            candidate_list=random.sample(candidate_list,12)
        for cnt, (x, y) in enumerate(data['ridge']["ridge_coordinate"]):
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
        # Call draw_square to visualize and save the image
        draw_square(
            image_path=data['enhanced_path'],
            points=selected_points,
            width=args.configs['crop_width'],
            left_text=f"ZhangYuan: {split_list[image_name]['zy']}",
            right_text=f"XieSJ: {split_list[image_name]['xie']}",
            save_path=os.path.join(visual_dir,image_name),  # Specify your save path format
            values=selected_values
        )
            
        # Aggregating results for the entire ima
        image_labels.append(label)
        image_predictions.append(predicted_label)  # Average probabilities

# Convert lists to arrays for metric calculations
image_labels = np.array(image_labels)
image_predictions = np.array(image_predictions)
acc = accuracy_score(image_labels, image_predictions)
auc = roc_auc_score(image_labels, image_predictions) 
print(f"Acc: {acc}, Auc: {auc}")