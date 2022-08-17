import copy
import numpy as np
import cv2
from detectron2.data import detection_utils
import torch
from pycocotools.coco import maskUtils

from detectron2.structures import BoxMode

from utils.custom_augmentations import Ax08

def custom_mask_mapper(dataset_dict,transform=Ax08(train_method='segm')):
    bbox_mode = BoxMode.XYWH_ABS
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    labels = [ann['category_id'] for ann in dataset_dict['annotations']]


    bboxes = [ann["bbox"] for ann in dataset_dict['annotations']]
    segmentations = [ann['segmentation'] for ann in dataset_dict['annotations']]

    masks = []
    for polygons in segmentations:
        for i in range(len(polygons)):
            polygons[i] = np.reshape(polygons[i], (-1, 2)).astype(np.int32)
        mask = np.zeros(image.shape[:2])
        cv2.fillPoly(mask, pts=polygons, color=255)
        masks.append(mask)

    args = {
        "image":image, 
        "category_id": labels,
        "masks": masks,
        "bboxes": bboxes,
        "keypoints": None, 
        "bbox_ids": np.arange(len(labels)),
    }
    args = {k: v for k, v in args.items() if v is not None}
    transformed = transform(**args)

    transformed_image = transformed["image"]
    transformed_labels = np.array(transformed['category_id'])
    transformed_bboxes = []
    transformed_masks = []
    
    visible_ids = transformed['bbox_ids']
    transformed_masks = np.array(transformed["masks"], dtype=np.uint8)[visible_ids]

    for i in range(len(transformed_masks)):
        rect = cv2.boundingRect(transformed_masks[i])
        transformed_bboxes.append(list(rect))

    dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))

    annos = []
    for i in range(len(transformed_labels)):
        anno = {'iscrowd': 0, 'category_id': transformed_labels[i], 'bbox_mode': bbox_mode}
        anno['bbox'] = transformed_bboxes[i]
        anno['segmentation'] = maskUtils.encode(np.asfortranarray(transformed_masks[i]))
        annos.append(anno)

    dataset_dict['annotations'] = annos

    instances = None
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")


    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

    return dataset_dict