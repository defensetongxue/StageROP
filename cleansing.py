from ridgeSegModule import generate_ridge
import os
import json
from utils_ import crop_square
import numpy as np
def select_points(point_list1, point_list2, l):
    selected_points = []
    for point1 in point_list1:
        for point2 in point_list2:
            if max(abs(point1[0]-point2[0]), abs(point1[1]-point2[1])) <= l:
                selected_points.append(point1)
                break
    return selected_points
def generate_crop(data_path,vessel_threhold=300):
    '''
        "image_name": image_name,
        "image_path":os.path.join(image_dict,image_name),
        "ridge_number": 0,
        "ridge_coordinate": [],
        "other_number": 0,
        "other_coordinate": [],
        "plus_number": 0,
        "plus_coordinate": [],
        "pre_plus_number": 0,
        "pre_plus_coordinate": [],
        "vessel_abnormal_number":0,
        "vessel_abnormal_coordinate":[],
        "class": label_class
    '''
    os.makedirs(os.path.join(data_path,'ridge_crop'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_crop')}/*")
    os.makedirs(os.path.join(data_path,'ridge_crop','annotations'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_crop',"vessel_crop"),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_crop',"image_crop"),exist_ok=True)

    for split in ['train','val','test']:
        with open(os.path.join(data_path,'ridge',f"{split}.json"),'r') as f:
            data_list=json.load(f)
        annotation=[]
        for data in data_list:
            ridge_list=data["ridge_coordinate"]
            vessel_list=data["vessel_abnormal_coordinate"]

            if data['class']==3:
                if data['ridge_number']>0 and data["vessel_abnormal_number"]<=0:
                    print(f"{data['image_name']} don't have vessel abnormal")
                    continue

                selected_points=select_points(point_list1= ridge_list,
                                           point_list2=vessel_list,
                                           l=vessel_threhold)
            elif data['class'] in [1,2]:
                selected_points=ridge_list
            else:
                raise ValueError(f"illegal class for ridge {data['class']}")
            cnt=0
            image_name=data['image_name'].split('.')[0]
            vessel_path=os.path.join(data_path,'vessel_seg',data['image_name'])
            heatmap_path=os.path.join(data_path,'ridge_mask',data['image_name'])
            for x,y in selected_points:
                crop_name=f"{image_name}_{str(cnt)}.jpg"
                cnt+=1
                vessel_crop_path=os.path.join(data_path,'ridge_crop','vessel_crop',crop_name)
                image_crop_path=os.path.join(data_path,'ridge_crop','image_crop',crop_name)
                crop_square(vessel_path,x,y,width=300,
                            save_path=vessel_crop_path)
                crop_square(data['image_path'],x,y,width=300,
                            save_path=image_crop_path)
                annotation.append({
                    "image_name":data['image_name'],
                    "image_path":data["image_path"],
                    "class":data["class"],
                    "crop_name":crop_name,
                    "crop_vessel_path":vessel_crop_path,
                    "crop_image_path":image_crop_path,
                    "heatmap_path":heatmap_path,
                })
        with open(os.path.join(data_path,'ridge_crop','annotations',f"{split}.json"),'w') as f:
            json.dump(annotation,f)
            
if __name__=='__main__':
    from config import get_config

    args=get_config()
    if args.generate_vessel:
        from VesselSegModule import generate_vessel_result
        generate_vessel_result(args.path_tar)
    if args.generate_ridge:
        generate_ridge(args.path_tar)
    generate_crop(args.path_tar)