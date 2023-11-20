import os,json
data_path='../autodl-tmp/dataset_ROP'
with open('./fin.json','r') as f:
    tmp=json.load(f)
unreapted=[]
repeat=[]
for image_name in tmp["sups"]:
    if image_name not in tmp["undefine"]:
        unreapted.append(image_name)
    else:
        repeat.append(image_name)
print(len(unreapted),len(repeat))
    