# python -u train.py --model_mode both --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 1 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 2 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 3 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 4  --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 5 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 6 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 7 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 8  --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 9 --save_name ./checkpoints/best.pth
python -u test_both.py  --model_mode both --test_crop_per_image 10 --save_name ./checkpoints/best.pth

python -u train.py --model_mode crop --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 1 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 2 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 3 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 4 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 5 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 6 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 7 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 8 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 9 --save_name ./checkpoints/crop.pth
python -u test_crop.py  --model_mode crop --test_crop_per_image 10  --save_name ./checkpoints/crop.pth

python -u train.py --model_mode heatmap  --save_name ./checkpoints/heatmap.pth
python -u test_heatmap.py  --model_mode heatmap  --save_name ./checkpoints/heatmap.pth
cd ../PerdictROP
python -u train.py 
python -u test.py