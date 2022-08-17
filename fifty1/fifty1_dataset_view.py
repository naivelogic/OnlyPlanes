import argparse, re
import fiftyone as fo
from fifty1_utils import fifty1_eval

def load_dataset(args):
    print("Loading image dataset from: ", args.data_path)
    
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset, # The type of the dataset being imported     
        name=args.dataset_name,
        label_types=args.label_type,
        labels_path=args.label_path,
        data_path=args.data_path,
        max_samples=args.max_samples, # None (default) if load whole dataset
        shuffle=args.shuffle,
    )
    return dataset

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_name', type=str, help="name of the dataset")
    parser.add_argument('--data_path', type=str, help="path of the dataset images directory")
    parser.add_argument('--label_path', type=str, help="path of the dataset label json")
    parser.add_argument('--max_samples', default=None, type=int, help="# of images to load")
    parser.add_argument('--shuffle', default=False, type=bool, help="shuffle the dataset")
    parser.add_argument('--segm_labels', action='store_true', help="set label_type to segmentations")  
    parser.add_argument('--eval_list',default=[], 
                        type=lambda s: re.split(',', s), #comma delimited list of characters
                        nargs='+', help="list of json results to viz evaluations")
    parser.add_argument('--delete_ds', action='store_true') #  if you want the default to be True then you could add an argument with action='store_false':
    parser.add_argument('--eval_view', action='store_true', help="view class wise FP analysis")
    # parse args
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()             # parse args
    #args=planes_datasets(args)
    print(args)
    if args.delete_ds:
        fo.delete_dataset(args.dataset_name)
    if args.segm_labels:
        args.label_type=("detections", "segmentations")
        #args.label_type=("segmentations")
    else:
        args.label_type=("detections")
    dataset = load_dataset(args)    # load dataset from dir
    
    if args.eval_list == []:
        # Ensures that the App processes are safely launched in local browser
        session = fo.launch_app(dataset, remote=True, port=5152)
        print(">>> dataset visualization complete")
    else:
        session = fifty1_eval(dataset, args.eval_list, args.eval_view, args.label_type)

    print("view dataset in by going to this url: http://localhost:5152/")
    session.wait()