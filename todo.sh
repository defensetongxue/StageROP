# python -u cleansing.py
python -u train_heatmap.py --save_name ./checkpoints/heatmap_vgg16.pth --cfg ./YAML/vgg16.yaml
python -u test_heatmap.py --save_name ./checkpoints/heatmap_vgg16.pth 
python -u train_both.py --save_name ./checkpoints/both_vgg16.pth --cfg ./YAML/vgg16.yaml --batch_size 4
python -u test_both.py --save_name ./checkpoints/both_vgg16.pth 
python -u train_crop.py --save_name ./checkpoints/crop_vgg16.pth --cfg ./YAML/vgg16.yaml
python -u test_crop.py --save_name ./checkpoints/crop_vgg16.pth

