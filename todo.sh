python -u cleansing.py
python -u train_crop.py --save_name ./checkpoints/crop_vgg16.pth --cfg ./YAML/vgg16.yaml
python -u train_crop.py --save_name ./checkpoints/crop_inception.pth --cfg ./YAML/inceptionv3.yaml
python -u train_heatmap.py --save_name ./checkpoints/heatmap_vgg16.pth --cfg ./YAML/vgg16.yaml
python -u train_heatmap.py --save_name ./checkpoints/heatmap_inception.pth --cfg ./YAML/inceptionv3.yaml
python -u train_both.py --save_name ./checkpoints/both_vgg16.pth --cfg ./YAML/vgg16.yaml
python -u train_both.py --save_name ./checkpoints/both_inception.pth --cfg ./YAML/inceptionv3.yaml

