import json
import os
from ridgeLocateModule import RidgeLocateProcesser
from PIL import Image
import random 

def crop_square(img_path, x, y, width,save_path=None):
    # Open the image file
    img = Image.open(img_path)

    # Calculate the top left and bottom right points of the square to be cropped
    left = x - width//2
    top = y - width//2
    right = left + width
    bottom = top + width

    # Crop the image and save it
    cropped_img = img.crop((left, top, right, bottom))
    if save_path:
        cropped_img.save(save_path)
    
    return cropped_img

def generate_crop_ridge_data(data_path,split='train',crop_width=200):
    with open(os.path.join(data_path,'ridge',f"{split}.json"),'r') as f:
        data=json.load(f)
    res_json=[]
    for annote in data:
        cnt=0
        for coord in annote["ridge_coordinate"]:
            x,y=coord
            new_name=f"{annote['image_name'].split('.')[0]}_{cnt}.jpg"
            crop_square(img_path=os.path.join(data_path,'images',annote['image_name']),x=x,y=y,
                        width=crop_width,
                        save_path=os.path.join(
                data_path,'crop_ridge_images',new_name))
            cnt+=1
            res_json.append({
                "image_path":os.path.join(data_path,'crop_ridge_images',new_name),
                "crop_from":os.path.join(data_path,'images',annote['image_name']),
                "class":annote["class"]
            })
    return res_json

def generate_train_val_data(data_path,crop_width,norm_images_ratio=3,crop_per_image=4):

    os.makedirs(os.path.join(data_path,'crop_ridge_images'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'crop_ridge_images')}/*")

    # generate abnormal_class samples
    train_annote=generate_crop_ridge_data(data_path,'train',crop_width)
    val_annote=generate_crop_ridge_data(data_path,'val',crop_width)

    # generate norm_class samples
    crop_processer=RidgeLocateProcesser(crop_per_image,20)
    with open(os.path.join(data_path, 'annotations', "train.json"),'r') as f:
        train_orignal_data=json.load(f)
    with open(os.path.join(data_path, 'annotations', "val.json"),'r') as f:
        val_orignal_data=json.load(f)
    train_norm_data=[i for i in train_orignal_data if i['class']==0]
    val_norm_data=[i for i in val_orignal_data if i['class']==0]
    train_norm_data=random.sample(
        train_norm_data,min(len(train_norm_data)-5,int(len(train_annote)*norm_images_ratio/crop_per_image)))
    val_norm_data=random.sample(
        val_norm_data,min(len(val_norm_data)-5,int(len(val_annote)*norm_images_ratio/crop_per_image)))

    for data in train_norm_data:
        preds,maxvals=crop_processer(data['image_path'])
        cnt=0
        for x,y in preds:
            new_name=f"{data['image_name'].split('.')[0]}_{str(cnt)}.jpg"
            crop_square(img_path=data["image_path"],x=x,y=y,
                        width=crop_width,
                        save_path=os.path.join(
                data_path,'crop_ridge_images',new_name))
            cnt+=1
            train_annote.append({
                "image_path":os.path.join(data_path,'crop_ridge_images',new_name),
                "crop_from":data["image_path"],
                "class":data["class"]
            })
    
    for data in val_norm_data:
        preds,maxvals=crop_processer(data['image_path'])
        cnt=0
        for x,y in preds:
            new_name=f"{data['image_name'].split('.')[0]}_{str(cnt)}.jpg"
            crop_square(img_path=data["image_path"],x=x,y=y,
                        width=crop_width,
                        save_path=os.path.join(
                data_path,'crop_ridge_images',new_name))
            cnt+=1
            val_annote.append({
                "image_path":os.path.join(data_path,'crop_ridge_images',new_name),
                "crop_from":data["image_path"],
                "class":data["class"]
            })
    
    # save
    os.makedirs(os.path.join(data_path,'crop_ridge_annotations'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'crop_ridge_annotations')}/*")

    with open(os.path.join(data_path,'crop_ridge_annotations','train.json'),'w') as f:
        json.dump(train_annote,f)
    with open(os.path.join(data_path,'crop_ridge_annotations','val.json'),'w') as f:
        json.dump(val_annote,f)

def generate_baseline_dataset(data_path,norm_images_ratio):
    os.makedirs(os.path.join(data_path,'crop_ridge_annotations_baseline'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'crop_ridge_annotations_baseline')}/*")

    with open(os.path.join(data_path,'crop_ridge_annotations','train.json'),'r') as f:
        train_annote=json.load(f)
    # Use a dict to remove duplicates based on the 'crop_from' key
    crop_from_dict = {item['crop_from']: {"class": item["class"],"image_path":item["crop_from"]} 
                       for item in train_annote}
    train_image_path_list=list(crop_from_dict.values())

    with open(os.path.join(data_path,'crop_ridge_annotations','val.json'),'r') as f:
        train_annote=json.load(f)
    # Use a dict to remove duplicates based on the 'crop_from' key
    crop_from_dict = {item['crop_from']: {"class": item["class"],"image_path":item["crop_from"]} 
                      for item in train_annote}
    val_image_path_list=list(crop_from_dict.values())

    with open(os.path.join(data_path,'crop_ridge_annotations_baseline','train.json'),'w') as f:
        json.dump(train_image_path_list,f)
    with open(os.path.join(data_path,'crop_ridge_annotations_baseline','val.json'),'w') as f:
        json.dump(val_image_path_list,f)

    # generate test baseline
    test_annote=[]
    with open(os.path.join(data_path,'ridge',"test.json"),'r') as f:
        test_data=json.load(f)
    for data in test_data:
        if data['ridge_number']<=0:
            continue
        test_annote.append({
            'image_path':os.path.join(data_path,'images',data['image_name']),
            'class':data['class']
        })
    
    # add norm for test data
    with open(os.path.join(data_path, 'annotations', "test.json"),'r') as f:
        test_orignal_data=json.load(f)
    test_norm_data=[i for i in test_orignal_data if i['class']==0]
    test_norm_data=random.sample(test_norm_data,int(len(test_annote)*norm_images_ratio))
    for data in test_norm_data:
        test_annote.append({
            'image_path':data['image_path'],
            'class':data['class']
        })
    with open(os.path.join(data_path,'crop_ridge_annotations_baseline','test.json'),'w') as f:
        json.dump(test_annote,f)

if __name__=='__main__':
    from config import get_config
    args=get_config()
    generate_train_val_data(args.path_tar,args.crop_width,args.norm_images_ratio,args.crop_per_image)
    generate_baseline_dataset(args.path_tar,args.norm_images_ratio)