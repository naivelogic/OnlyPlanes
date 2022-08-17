import detectron2.data.transforms as T
import copy
from detectron2.data import detection_utils
import torch

from detectron2.structures import BoxMode

from utils.custom_augmentations import Ax08

def custom_bbox_mapper(dataset_dict):
    Amin_size = (800,) 
    Amax_size = 1333 
    Asample_style = "choice" 
    dataset_dict = copy.deepcopy(dataset_dict)  
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

    transform = Ax08(train_method='bbox')

    anns = [ann['bbox'] + [ann['category_id']] for ann in dataset_dict['annotations']]
    tr_output = transform(image=image, bboxes=anns)
    image = tr_output['image']
    height, width, _ =image.shape
    anns = tr_output['bboxes']
    anns = [{'category_id':ann[-1], 'bbox':ann[:-1], 
             'bbox_mode':BoxMode.XYWH_ABS} for ann in anns]
    auginput = T.AugInput(image)
    
    transform = T.ResizeShortestEdge(Amin_size, Amax_size, Asample_style)(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))

    annos = [
            detection_utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in anns
        ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[1:])
                                   
    return {
        "file_name":dataset_dict["file_name"],
        "image_id": dataset_dict["image_id"],
        "height": height,
        "width": width,
        "image": image,
        "instances": detection_utils.filter_empty_instances(instances)
        }
        