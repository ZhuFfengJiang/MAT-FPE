# ------------------ Dataset Config ------------------
from .data_config.dataset_config import dataset_cfg


def build_dataset_config(args):
    if args.dataset in ['coco', 'coco-val', 'coco-test']:
        cfg = dataset_cfg['coco']
    else:
        cfg = dataset_cfg[args.dataset]

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg


# ------------------ Transform Config ------------------
from .data_config.transform_config import (
    # YOLOv5-Style
    yolov5_p_trans_config,
    yolov5_n_trans_config,
    yolov5_s_trans_config,
    yolov5_m_trans_config,
    yolov5_l_trans_config,
    yolov5_x_trans_config,
    # YOLOX-Style
    yolox_p_trans_config,
    yolox_n_trans_config,
    yolox_s_trans_config,
    yolox_m_trans_config,
    yolox_l_trans_config,
    yolox_x_trans_config,
    # SSD-Style
    ssd_trans_config,
)

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
   
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config

    # YOLOv5-style transform 
    elif trans_config == 'yolov5_p':
        cfg = yolov5_p_trans_config
    elif trans_config == 'yolov5_n':
        cfg = yolov5_n_trans_config
    elif trans_config == 'yolov5_s':
        cfg = yolov5_s_trans_config
    elif trans_config == 'yolov5_m':
        cfg = yolov5_m_trans_config
    elif trans_config == 'yolov5_l':
        cfg = yolov5_l_trans_config
    elif trans_config == 'yolov5_x':
        cfg = yolov5_x_trans_config
        
    # YOLOX-style transform 
    elif trans_config == 'yolox_p':
        cfg = yolox_p_trans_config
    elif trans_config == 'yolox_n':
        cfg = yolox_n_trans_config
    elif trans_config == 'yolox_s':
        cfg = yolox_s_trans_config
    elif trans_config == 'yolox_m':
        cfg = yolox_m_trans_config
    elif trans_config == 'yolox_l':
        cfg = yolox_l_trans_config
    elif trans_config == 'yolox_x':
        cfg = yolox_x_trans_config

    print('Transform Config: {} \n'.format(cfg))

    return cfg


# ------------------ Model Config ------------------
## YOLO series

from .model_config.yolov8_config import yolov8_cfg
def build_model_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))

    # YOLOv8
    if args.model in ['yolov8_n', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x']:
        cfg = yolov8_cfg[args.model]
   
    return cfg

