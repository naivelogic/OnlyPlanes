#!/bin/bash
#source ~/anaconda3/bin/activate OP
#conda activate OP

DATASET_NAME=name_of_dataset                #(e.g., rareplanesTest)
DS_IMG_DIR=/path/to/image/directory         #(e.g., rareplanes/test/PS-RGB_tiled)
DS_LABEL_DIR=/path/to/coco_annotations.json #(e.g., rareplanes/test/aircraft_real_test_coco.json)
EVAL_LIST1=common_model_name,path/to/coco_instances_results.json
EVAL_LIST2=common_model_name,path/to/coco_instances_results.json

python fifty1_dataset_view.py --dataset_name ${DATASET_NAME} \
                              --data_path ${DS_IMG_DIR} --label_path ${DS_LABEL_DIR} \
                              --max_samples 50 --shuffle True \
                              --eval_list ${EVAL_LIST1} ${EVAL_LIST2}




