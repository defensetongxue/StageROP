cd ../ROP_diagnose
python -u cleansing.py
cd ../ridge_location
python -u cleansing.py
python -u train.py
python -u test.py
cd ../StageROP
python -u cleansing.py
python -u train.py
python -u ring.py