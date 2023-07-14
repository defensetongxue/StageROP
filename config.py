import argparse
from yacs.config import CfgNode as CN


_C = CN()


_C.GPUS = (0, )
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.NUM_CLASS=3
_C.RESULT_PATH= './experiments'
_C.IMAGE_SIZE=[1600,1200]
_C.IMAGE_RESIZE = [800, 800]  # width * height

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAINSET = ''
_C.DATASET.TESTSET = ''

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.25
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 300

_C.TRAIN.EARLY_STOP = 30
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_tar', type=str, default='../autodl-tmp/dataset_ROP',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--crop_width', type=int, default=100,
                        help='crop width.')
    parser.add_argument('--crop_per_image', type=int, default=8,
                        help='crop_per_image.')
    parser.add_argument('--generate_vessel', type=bool, default=True,
                        help='if generate vesel.')
    # Model
    parser.add_argument('--model_mode', type=str, default="heatmap",
                        help='Path to the source folder containing original datasets.')
    # train and test
    parser.add_argument('--save_name', type=str, default="./checkpoints/best.pth",
                        help='Name of the file to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="experiments",
                        help='Path to the visualize result or the pytorch model will be saved.')
    parser.add_argument('--from_checkpoint', type=str, default="",
                        help='load the exit checkpoint.')
    
    # config file 
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./YAML/vgg16.yaml", type=str)
    # test
    parser.add_argument('--test_crop_per_image', help='test_crop_per_image',
                        default=8, type=int)
    parser.add_argument('--test_crop_distance', help='test_crop_distance',
                        default=20, type=int)
    args = parser.parse_args()
    # Merge args and config file 
    update_config(_C, args)
    args.configs=_C

    return args