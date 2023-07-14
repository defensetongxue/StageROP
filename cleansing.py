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
    for split in ['train','val','test']:
        with open(os.path.join(data_path,'ridge',f"{split}.json"),'r') as f:
            data_list=json.load(f)
        for data in data_list:
            if data['class']==3:
                if data['ridge_number']>0 and data["vessel_abnormal_number"]<=0:
                    print(f"{data['image_name']} don't have vessel abnormal")
                    continue
                
                ridge_list=data["ridge_coordinate"]
                vessel_list=data["vessel_abnormal_coordinate"]
                selected_points=select_points(point_list1= ridge_list,
                                           point_list2=vessel_list,
                                           l=vessel_threhold)
                for x,y in selected_points:
                    crop_square(data['image_path'],x,y,width=300,
                                visual_path='./point.jpg',
                                save_path='./crop.jpg')
                    raise
            elif data['class'] in [1,2]:
                continue
            else:
                raise ValueError("illegal class for ridge")
if __name__=='__main__':
    from config import get_config

    args=get_config()
    if args.generate_vessel:
        from VesselSegModule import generate_vessel_result
        generate_vessel_result(args.path_tar)
    # generate_ridge(args.path_tar)
    # generate_crop(args.path_tar)