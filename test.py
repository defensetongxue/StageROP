import json
import os
import torch
from torchvision import transforms
from config import get_config
from utils_ import to_device, crop_square
from models import build_model
import numpy as np
from PIL import Image
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score

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
with open(os.path.join(args.data_path, 'split', f'{args.split_name}.json'), 'r') as f:
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
# def decide_strategy(predicted_probs):
#     # Sum probabilities across all crops for each class to get the mass
#     summed_probs = np.sum(predicted_probs, axis=0)
#     # Use max summed probability to determine label
#     return np.argmax(summed_probs)
# Function to calculate accuracy and AUC
def calculate_metrics(true_labels, predicted_labels, predicted_probs):
    acc = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovo')  # Use 'ovr' for One-vs-Rest
    return acc, auc

# Initialize lists to hold the aggregated results
image_labels = []
image_predictions = []
image_probabilities = []

with torch.no_grad():
    for image_name in split_list:
        data = data_dict[image_name]
        label = data['stage']
        if label==0 or label>3:
            continue
        if 'ridge' not in data:
            # print(image_name)
            continue
        label-=1
        crop_img_list = []
        crop_probs = []  # List to store probabilities of each cropped image
        
        # Process each coordinate to get the cropped image and its prediction
        for cnt, (x, y) in enumerate(data['ridge']["ridge_coordinate"]):
            crop_img = crop_square(data['enhanced_path'], x, y, args.configs['crop_width']).convert('RGB')
            processed_img = img_process(crop_img).unsqueeze(0).to(device)
            output = model(processed_img).cpu()
            probs = torch.softmax(output, dim=1)
            crop_probs.append(probs.squeeze().numpy())

        # Aggregating results for the entire image
        crop_probs = np.array(crop_probs)
        image_labels.append(label)
        predicted_label = decide_strategy(crop_probs)
        image_predictions.append(predicted_label)
        image_probabilities.append(crop_probs.mean(axis=0))  # Average probabilities

# Convert lists to arrays for metric calculations
image_labels = np.array(image_labels)
image_predictions = np.array(image_predictions)
image_probabilities = np.array(image_probabilities)

# Calculate metrics
accuracy, auc_score = calculate_metrics(image_labels, image_predictions, image_probabilities)
def calculate_one_hot_auc(true_labels, predicted_labels, num_classes):
    # One-hot encode the predicted labels
    one_hot_predictions = label_binarize(predicted_labels, classes=range(num_classes))
    # Calculate AUC
    one_hot_auc = roc_auc_score(true_labels, one_hot_predictions, multi_class='ovr')
    return one_hot_auc
one_hot_auc_score = calculate_one_hot_auc(image_labels, image_predictions, 3)

print(f"Image-level Accuracy: {accuracy:.6f}")
print(f"Image-level AUC: {auc_score:.6f}")
print(f"one hot AUC:{one_hot_auc_score:.6f}")
print("Finished testing!")
