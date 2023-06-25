from .ridge_segmentation_prcesser import ridge_segmentation_processer
import os 
import json
def generate_ridge(data_path):
    splits=['train','val','test']
    os.makedirs(os.path.join(data_path,'ridge_mask'),exist_ok=True)
    processer=ridge_segmentation_processer('point',5,50)
    for split in splits:
        with open(os.path.join(data_path,'annotations',f'{split}.json'),'r') as f:
            data_list=json.load(f)
        annotation_res=[]
        for data in data_list:
            _,annotes=processer(data['image_path'],
                                save_path=os.path.join(
                data_path,'ridge_mask',data['image_name'].split('.')[0]+'pt'))