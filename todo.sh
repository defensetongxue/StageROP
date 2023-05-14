python -u cleansing.py --crop_width 100
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 10
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py --test_crop_per_image 5 --test_crop_distance 10
python -u test.py --test_crop_per_image 5  --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 10
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 10
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20
python -u test.py  --test_crop_per_image 15 --test_crop_distance 10
python -u test.py  --test_crop_per_image 15 --test_crop_distance 20
cd ../PerdictROP
python -u train.py
python -u test.py
cd ../StageROP 
python -u ring.py

python -u cleansing.py --crop_width 50
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 10
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py --test_crop_per_image 5 --test_crop_distance 10
python -u test.py --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 10
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 10
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20
python -u test.py  --test_crop_per_image 15 --test_crop_distance 10
python -u test.py  --test_crop_per_image 15 --test_crop_distance 20
cd ../PerdictROP
python -u train.py
python -u test.py
cd ../StageROP 
python -u ring.py

python -u cleansing.py --crop_width 150
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 10
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py --test_crop_per_image 5 --test_crop_distance 10
python -u test.py --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 10
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 10
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20
python -u test.py  --test_crop_per_image 15 --test_crop_distance 10
python -u test.py  --test_crop_per_image 15 --test_crop_distance 20
cd ../PerdictROP
python -u train.py
python -u test.py
cd ../StageROP 
python -u ring.py


python -u cleansing.py --crop_width 200
python -u train.py
python -u test.py --test_crop_per_image 3 --test_crop_distance 10
python -u test.py --test_crop_per_image 3 --test_crop_distance 20
python -u test.py --test_crop_per_image 5 --test_crop_distance 10
python -u test.py --test_crop_per_image 5 --test_crop_distance 20
python -u test.py --test_crop_per_image 8 --test_crop_distance 10
python -u test.py --test_crop_per_image 8 --test_crop_distance 20
python -u test.py  --test_crop_per_image 10 --test_crop_distance 10
python -u test.py  --test_crop_per_image 10 --test_crop_distance 20
python -u test.py  --test_crop_per_image 15 --test_crop_distance 10
python -u test.py  --test_crop_per_image 15 --test_crop_distance 20
cd ../PerdictROP
python -u train.py
python -u test.py
cd ../StageROP 
python -u ring.py
