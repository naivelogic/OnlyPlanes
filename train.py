#!/usr/bin/python
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from detectron2.config import *
from utils.custom_trainers import Trainer_bbox, Trainer_mask
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_setup
import logging
logger = logging.getLogger("detectron2")


def main(args):
    TRAIN_CONFIG = args.config_file
    OUTPUT_PATHS = args.output_folder
    
    TRAIN_IMG_DIR = args.train_img_dir
    TRAIN_COCO_JSON = args.train_coco_json
    VAL_IMG_DIR= args.val_img_dir
    VAL_COCO_JSON = args.val_coco_json

    register_coco_instances(f"custom_dataset_train", {},TRAIN_COCO_JSON , TRAIN_IMG_DIR)

    register_coco_instances(f"custom_dataset_val", {},VAL_COCO_JSON , VAL_IMG_DIR,) 


    cfg = get_cfg()
    cfg.merge_from_file(TRAIN_CONFIG) #TRAINING_MODEL_YAML
    cfg.OUTPUT_DIR= OUTPUT_PATHS 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #lets just check our output dir exists

    cfg.freeze()                    # make the configuration unchangeable during the training process
    default_setup(cfg, args)
    cfg.dump()
    
    if args.train_method == 'bbox':
        trainer = Trainer_bbox(cfg)
    elif args.train_method == 'segm':
        trainer = Trainer_mask(cfg)
    ## TODO: add rotated and obb
    else:
      print("define train_method in arugments")
      exit()

    trainer.resume_or_load()
    trainer.train()

if __name__ == "__main__":
    from utils.args_lib import local_train_args
    args = local_train_args().parse_args()
    
    main(args)
    
