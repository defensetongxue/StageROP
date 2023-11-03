import os
import json
from utils_ import crop_square
from PIL import Image,ImageDraw
def select_points(point_list1, point_list2, l):
    # for stage 3, we only select those ridge points nearby the vessel abnormal
    selected_points = []
    for point1 in point_list1:
        for point2 in point_list2:
            if max(abs(point1[0]-point2[0]), abs(point1[1]-point2[1])) <= l:
                selected_points.append(point1)
                break
    return selected_points
def visual_square(image_path, selected_points, box_width, save_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Half of the box width to calculate the box coordinates
    half_box_width = box_width // 2

    # Draw a box for each point
    for point in selected_points:
        # Calculate coordinates of the top left and bottom right corners
        top_left = (point[0] - half_box_width, point[1] - half_box_width)
        bottom_right = (point[0] + half_box_width, point[1] + half_box_width)

        # Draw a red rectangle with thickness of 10
        draw.rectangle([top_left, bottom_right], outline='red', width=10)

    # Save the image
    image.save(save_path)

def generate_crop(data_path,vessel_threhold=300,crop_width=300):
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
                
                visual_square(data['vessel_path'],selected_points,crop_width,save_path='./experiments/visual_annotation/'+image_name[:-4]+'_v.jpg')
                visual_square(data['enhanced_path'],selected_points,crop_width,save_path='./experiments/visual_annotation/'+image_name)
            # elif data['stage'] in [1,2]:
            #     seleczted_points=data_ridge["ridge_coordinate"]
            # else:
            #     print(image_name,' ',data['stage'])
            #     continue
            
        
    with open(os.path.join(data_path,'stage_rop','crop_annotations.json'),'w') as f:
        json.dump(annotation_crop,f)
if __name__ =='__main__':
    generate_crop('../autodl-tmp/dataset_ROP')