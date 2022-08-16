#!/bin/bash
# This script downloads the OnlyPlanes 1.0 models for object detection and instance segmentation. 

mkdir -p models
cd models

SRC=https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22

wget --no-check-certificate ${SRC}/onlyplanes_faster_rcnn_r50-0034999.pth ./  # binary object detection model
wget --no-check-certificate ${SRC}/onlyplanes_faster_rcnn_r50-config.yaml ./  # binary object detection config

wget --no-check-certificate ${SRC}/onlyplanes_mask_rcnn_r50-0024999.pth ./    # instance segmentation model
wget --no-check-certificate ${SRC}/onlyplanes_mask_rcnn_r50-config.yaml ./    # instance segmentation config
