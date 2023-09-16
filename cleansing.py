from ridgeSegModule import generate_ridge
import os
import json
from utils_ import crop_square
import numpy as np
def select_points(point_list1, point_list2, l):
    # for stage 3, we only select those ridge points nearby the vessel abnormal
    selected_points = []
    for point1 in point_list1:
        for point2 in point_list2:
            if max(abs(point1[0]-point2[0]), abs(point1[1]-point2[1])) <= l:
                selected_points.append(point1)
                break
    return selected_points

def generate_crop(data_path,vessel_threhold=300,crop_width=300):
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
    os.makedirs(os.path.join(data_path,'stage_rop'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'stage_rop')}/*")
    os.makedirs(os.path.join(data_path,'stage_rop','crop_split'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'stage_rop',"vessel_crop"),exist_ok=True)
    os.makedirs(os.path.join(data_path,'stage_rop',"image_crop"),exist_ok=True)

    with open(os.path.join(data_path,"annotations.json"),'r') as f:
        data_list=json.load(f)
    annotation_crop={}
    for image_name in data_list:
        data=data_list[image_name]
        if 'ridge' in data:
            data_ridge=data['ridge']
            if data['stage']==3:
                if data_ridge['ridge_number']>0 and data_ridge["vessel_abnormal_number"]<=0:
                    # in this condition, anntation make mistakes
                    print(f"{data['image_name']} don't have vessel abnormal")
                    continue

                selected_points=select_points(point_list1= data_ridge["ridge_coordinate"],
                                           point_list2=data_ridge["vessel_abnormal_coordinate"],
                                           l=vessel_threhold)
            elif data['stage'] in [1,2]:
                selected_points=data_ridge["ridge_coordinate"]
            else:
                raise ValueError(f"illegal stage for ridge {data['stage']}")
            cnt=0
            for x,y in selected_points:
                crop_name=f"{data['id']}_{str(cnt)}.jpg"
                cnt+=1
                vessel_crop_path=os.path.join(data_path,'stage_rop','vessel_crop',crop_name)
                image_crop_path=os.path.join(data_path,'stage_rop','image_crop',crop_name)
                crop_square(data['vessel_path'],x,y,width=crop_width,
                            save_path=vessel_crop_path)
                crop_square(data['image_path'],x,y,width=crop_width,
                            save_path=image_crop_path)
                annotation_crop[crop_name]={
                    "crop_from":image_name,
                    "stage":data["stage"],
                    "crop_vessel_path":vessel_crop_path,
                    "crop_image_path":image_crop_path
                }
        
    with open(os.path.join(data_path,'stage_rop','crop_annotations.json'),'w') as f:
        json.dump(annotation_crop,f)

def generate_crop_split(data_path,split_name):
    os.makedirs(os.path.join(data_path,'stage_rop','split'),exist_ok=True)
    with open(os.path.join(data_path,'split',f"{split_name}.json"),'r') as f:
        split_orignal=json.load(f)
    with open(os.path.join(data_path,'stage_rop','crop_annotations.json'),'r') as f:
        crop_annotation=json.load(f)
    split_dict={}
    for split in ['train','val','test']:
        for image_name in split_orignal[split]:
            split_dict[image_name.split('.')[0]]=split
    split_new={'train':[],'val':[],'test':[]}
    for crop_name in crop_annotation:
        tar_split=split_dict[crop_name.split('_')[0]]
        split_new[tar_split].append(crop_name)
    with open(os.path.join(data_path,'stage_rop','split',f"{split_name}.json"),'w') as f:
        json.dump(split_new,f)

if __name__=='__main__':
    from config import get_config

    args=get_config()
    
    if args.generate_crop:
        generate_crop(args.data_path,
                      vessel_threhold=args.configs['vessel_disctance_threshold'],
                      crop_width=args.configs['crop_width'])
    if args.generate_split:
        generate_crop_split(args.data_path,split_name=args.split_name)