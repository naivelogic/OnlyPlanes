#!/bin/bash
#source ~/anaconda3/bin/activate OP
#conda activate OP

DATASET_NAME=name_of_dataset                #(e.g., OnlyPlanes)
DS_IMG_DIR=/path/to/image/directory         #(e.g., datasets/OnlyPlanes/images)
DS_LABEL_DIR=/path/to/coco_annotations.json #(e.g., datasets/OnlyPlanes/coco_ds/OnlyPlanes_binary_plane_annotations.json)

python fifty1_dataset_view.py --dataset_name ${DATASET_NAME} \
                              --data_path ${DS_IMG_DIR} --label_path ${DS_LABEL_DIR} \
                              --max_samples 50 --segm_labels --shuffle True


