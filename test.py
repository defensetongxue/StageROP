import json
import os
import torch
from cleansing import crop_square
from torchvision import transforms
from config import get_config
from utils_ import get_instance,ContrastEnhancement,sensitive_score
import models
from sklearn.metrics import accuracy_score
from ridgeLocateModule import RidgeLocateProcesser
import numpy as np
import cv2
def visualize_and_save_landmarks(image_path, 
                                 preds, maxvals, save_path,text=False):
    print(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure preds and maxvals are NumPy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.squeeze(0).numpy()
    if isinstance(maxvals, torch.Tensor):
        maxvals = maxvals.squeeze().numpy()
    # Draw landmarks on the image
    cnt=1
    for pred, maxval in zip(preds, maxvals):
        x, y = pred
        # x,y=x*w_r,y*h_r
        cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), -1)
        if text:
            cv2.putText(img, f"{maxval:.2f}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"{cnt}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cnt+=1
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return preds,maxvals

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
crop_per_image=args.test_crop_per_image
crop_distance=args.test_crop_distance
crop_processer=RidgeLocateProcesser(crop_per_image,crop_distance)
all_targets = []
all_outputs = []
test_transforms=transforms.Compose([
                ContrastEnhancement(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4485, 0.5278, 0.5477], std=[0.0910, 0.1079, 0.1301])
            ])

print({f"teat name number {len(test_data_list)}"})
with torch.no_grad():
    for test_data in test_data_list:
        image_path=test_data['image_path']
        label=test_data['class']
        preds,maxvals=crop_processer(image_path)
        cnt=0
        predict_labels_image=0
        # visualize_and_save_landmarks(image_path,preds,maxvals,os.path.join('./experiments/visual',os.path.basename(image_path)))
        for x,y in preds:
            crop_image=crop_square(image_path,x=x,y=y,
                        width=args.crop_width,
                        # save_path='./experiments/crop/'+os.path.basename(image_path).split('.')[0]+f'_{str(cnt)}.jpg'
                        )
            img=test_transforms(crop_image).to(device)
            outputs = model(img.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1).squeeze().cpu()
            predict_labels_image=max(predict_labels_image,int(predicted_labels))
            cnt+=1
        all_targets.append(label)
        all_outputs.append(predict_labels_image)
acc = accuracy_score(np.array(all_targets),np.array(all_outputs))
sens=sensitive_score(all_targets,all_outputs,test_data_list)
print(f"crop_per_image: {crop_per_image} crop_distance: {crop_distance}")
print(f"Finished testing! Test acc {acc:.4f} sensitive: {(sens):.4f}")
