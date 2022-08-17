import argparse
import sys, os

def test_tron_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='normal', required=False)
    parser.add_argument("--output_dir", type=str, default="" , required=False)
    parser.add_argument("--conf", type=float, default=0.5, required=False)
    parser.add_argument("--use_gpu", type=bool, default=False, required=False)
    parser.add_argument("--ckpt_path", type=str, default="", required=False)
    parser.add_argument("--num_images", type=int, default=5, required=False)
    return parser


def local_train_args(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    
    parser = argparse.ArgumentParser(epilog=epilog,formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument('--output-folder', type=str, dest='output_folder', help='trained model folder mounting point')
    parser.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    
    parser.add_argument('--train_img_dir', type=str, dest='train_img_dir', help='training image data folder mounting point')
    parser.add_argument('--train_coco_json', type=str, dest='train_coco_json', help='training data annotation json file')
    parser.add_argument('--val_img_dir', type=str, dest='val_img_dir', help='training image data folder mounting point')
    parser.add_argument('--val_coco_json', type=str, dest='val_coco_json', help='training data annotation json file')
    parser.add_argument('--train_method', type=str, dest='train_method', help='training methods: bbox, segm')

    return parser