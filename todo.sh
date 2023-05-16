python -u cleansing.py --crop_width 200
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py  --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20

python -u cleansing.py --crop_width 150
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py  --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20


python -u cleansing.py --crop_width 250
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py  --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20

