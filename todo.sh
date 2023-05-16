python -u train.py --model_mode both
python -u test.py  --model_mode both --test_crop_per_image 1
python -u test.py  --model_mode both --test_crop_per_image 2
python -u test.py  --model_mode both --test_crop_per_image 3 
python -u test.py  --model_mode both --test_crop_per_image 4 
python -u test.py  --model_mode both --test_crop_per_image 5 
python -u test.py  --model_mode both --test_crop_per_image 6
python -u test.py  --model_mode both --test_crop_per_image 7
python -u test.py  --model_mode both --test_crop_per_image 8 
python -u test.py  --model_mode both --test_crop_per_image 9
python -u test.py  --model_mode both --test_crop_per_image 10

python -u train.py --model_mode crop
python -u test.py  --model_mode crop --test_crop_per_image 1
python -u test.py  --model_mode crop --test_crop_per_image 2
python -u test.py  --model_mode crop --test_crop_per_image 3 
python -u test.py  --model_mode crop --test_crop_per_image 4 
python -u test.py  --model_mode crop --test_crop_per_image 5 
python -u test.py  --model_mode crop --test_crop_per_image 6
python -u test.py  --model_mode crop --test_crop_per_image 7
python -u test.py  --model_mode crop --test_crop_per_image 8 
python -u test.py  --model_mode crop --test_crop_per_image 9
python -u test.py  --model_mode crop --test_crop_per_image 10

python -u train.py --model_mode heatmap
python -u test.py  --model_mode heamap 