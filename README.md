# OnlyPlanes: Incrementally Tuning Synthetic Training Datasets for Satellite Object Detection
Project Links:  [[Paper][paper_link]] [[Project Page][project_page]] [[Video][youtube_vid]] [[Blog][medium_blog_series]]

***Abstract:*** _This paper addresses the challenge of solely using synthetic data to train and improve computer vision models for detecting airplanes in satellite imagery by iteratively tuning the training dataset. The domain gap for computer vision algorithms remains a continued struggle to generalize when learning only from synthetic images and produce results comparable to models learning from real images. We present OnlyPlanes, a synthetic satellite image training dataset of airplanes in a top-view that contains 12,500 images, 132,967 aircraft instances, with 80 fine grain attributes. We detail the synthetic rendering process to procedurally-generate and render training images and introduce the synthetic learning feedback loop that iteratively finetunes the training dataset to improve model performance. Experiments show the prior performance degradations and false positives when testing the model on real-world images were corrected when incrementally incorporating new features to the training dataset such as diverse plane formations and altitude variations. We demonstrate the promising future for continually improving models by tuning the synthetic training data developed in an iterative fashion to effectively close the synthetic to real domain gap. The OnlyPlanes dataset, source code and trained models are available at https://github.com/naivelogic/OnlyPlanes._

[paper_link]: docs/OnlyPlanes_report_placeholder.pdf
[project_page]: https://naivelogic.github.io/OnlyPlanes/
[medium_blog_series]: TBD
[youtube_vid]: TBD

### Example Inference Results from OnlyPlanes Models tested on Real-World Datasets

![](docs/media/OnlyPlanes_example_inferences_real_datasets.png)

## Dataset Overview
OnlyPlanes is a small dataset (less than 50k images and less than 20GB in weight) for airplane object and instant segmentation computer vision task from satellite imagery.

| Dataset                               	| Size      	| Description 	|
|---------------------------------------	|-----------	|-------------	|
| [OnlyPlanes.zip][ds1]                 	| 14.8 GB   	| Training    	|
| [binary_plane_annotations.json][lb01] 	| 1.57 GB   	| Labels      	|
| [civil_role_annotations.json][lb02]   	| 506.43 MB 	| Labels      	|
| [Bing2D_Empty.zip][ds2]               	| 531.8 MB  	| Training    	|
| [Bing2Dempty_annotations.json][lb03]  	| 0.137 MB  	| Labels      	|

[ds1]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/OnlyPlanes_dataset_08122022.zip
[lb01]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/OnlyPlanes_binary_plane_annotations_imgdir.json
[lb02]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/civil_role_annotations.json
[ds2]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/bing2d_empty/Bing2D_empty_airports.zip
[lb03]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/bing2d_empty/coco_annotations_emptyBing2Dairport_1024split_small.json



<details>
 <summary>Dataset Description</summary>
The OnlyPlanes dataset contains 12,500 images and 132,967 instance objects consisting of four categories (plane, jumbo jet, military, helicopter) with 80 fine-grain attributes that define the plane model (e.g., Boeing 737). A single training dataset is provided for both object detection and instance segmentation tasks at 1024x1024 image resolution using ten different airport. 

![](docs/media/OnlyPlanes_Categories.png)
</details>


## Models Overview
The training methodology implemented involved a Faster-RCNN and a Mask-RCNN model solely trained on the OnlyPlanes synthetic dataset. The Faster-RCNN model is used for object detection and the Mask-RCNN model is used for instance segmentation. 

| Name                             | Model Zoo Config               | Trained Model                                | Train Config       |
|----------------------------------|--------------------------------|----------------------------------------------|--------------------|
| OnlyPlanes Object Detection      | [faster_rcnn_R_50_FPN_3x][m01] | [onlyplanes_faster_rcnn_r50-0034999.pth][m0] | [config.yaml][m02] |
| OnlyPlanes Instance Segmentation | [mask_rcnn_R_50_FPN_3x][m03]   | [onlyplanes_mask_rcnn_r50-0054999.pth][m04]  | [config.yaml][m05] |

[m01]: https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
[m0]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22/onlyplanes_faster_rcnn_r50-0034999.pth
[m02]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22/onlyplanes_faster_rcnn_r50-config.yaml
[m03]: https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
[m04]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22/onlyplanes_mask_rcnn_r50-0024999.pth
[m05]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22/onlyplanes_mask_rcnn_r50-config.yaml


### Benchmarking

To evaluate the performance of OnlyPlanes we tested the performance on detecting airplanes in other notable datasets. Benchmark datasets include: RarePlanes Real Test, DIOR Test, iSAID Val, and NWPU VHR-10.

| Benchmark Dataset |  mAP  | mAP50 |   AR  | OnlyPlanes Model |
|-------------------|:-----:|:-----:|:-----:|------------------|
| RarePlanes        | 59.10 | 91.10 | 65.40 | Faster R-CNN     |
| NWPU VHR10        | 73.66 | 98.32 | 78.90 | Faster R-CNN     |
| NWPU VHR10        | 42.53 | 98.25 | 47.61 | Mask R-CNN       |
| DIOR              | 48.56 | 82.73 | 57.48 | Faster R-CNN     |
| iSAID             | 46.48 | 68.99 | 53.89 | Faster R-CNN     |
| iSAID             | 20.56 | 57.74 | 30.25 | Mask R-CNN       |

<details>
 <summary>Download and Prepare Benchmark Datasets</summary>

To test the performance of the model the below benchmark datasets were used. 

* iSAID | [paper][isaid_paper] | [dataset][isaid_ds] | [binary plane json][isaid_ds]
  * Note: We used the pre-processed version of the dataset from [CATNet](https://github.com/yeliudev/CATNet) approach where images were split into 512 x 512 patches and extreme aspect ratios from the official toolkit were corrected.
  * Test dataset statistics: 11,752 images | 6,613 airplane instances from the validation dataset. 

  [isaid_paper]: https://arxiv.org/abs/1905.12886
  [isaid_ds]: https://connectpolyu-my.sharepoint.com/personal/21039533r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FCATNet%2Fisaid%5Fpatches%2D85c7fca6%2Ezip&parent=%2Fpersonal%2F21039533r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FZoo%2FReleases%2FCATNet&ga=1
  [isaid_json_ours]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/test_ds_binary_json/isaid_val_binary_plane_coco_annotations_ALL.json

* RarePlanes | [paper][RarePlanes_paper] | [dataset][RarePlanes_ds] | [binary plane json][RarePlanes_json_ours]
  * Note: for evaluation only RarePlanes Real Test dataset was used. Instructions to download the RarePlanes dataset [RarePlanes Public User Guide](https://www.cosmiqworks.org/rareplanes-public-user-guide/). Additionally, since the [official RarePlanes repository](https://github.com/aireveries/RarePlanes) is no longer available (summer 2022), refer to the [unofficial mirror repo](https://github.com/VisionSystemsInc/RarePlanes).
  * Test dataset statistics: 2,710 images and 6,812 airplane instances from the real test dataset.

  [RarePlanes_paper]: https://arxiv.org/abs/2006.02963
  [RarePlanes_ds]: https://www.cosmiqworks.org/rareplanes-public-user-guide/
  [RarePlanes_json_ours]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/test_ds_binary_json/rareplanes_aircraft_real_test_coco_ph.json

* NWPU VHR10 | [paper][nwpu10_paper] | [dataset][nwpu10_ds] | [binary plane json][nwpu10_json_ours]
  * Test dataset statistics: 650 images and 757 airplane instances from the positive image set. 

  [nwpu10_paper]: https://arxiv.org/abs/2006.02963
  [nwpu10_ds]: https://1drv.ms/u/s!AmgKYzARBl5cczaUNysmiFRH4eE
  [nwpu10_json_ours]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/test_ds_binary_json/nwpu_vhr10_binary_plane_coco_annotations_ALL.json


* DIOR | [paper][DIOR_paper] | [dataset][DIOR_ds] | [binary plane json][DIOR_json_ours]
  * Note: we first converted the DIOR VOC to COCO annotations using only the we used the `test.txt` Horizontal Bounding Boxes annotations.  
  * Test dataset statistics: 2,932 images and 8,042 airplane instances from the test dataset.

  [DIOR_paper]: https://arxiv.org/abs/1909.00133
  [DIOR_ds]: https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC
  [DIOR_json_ours]: https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/test_ds_binary_json/DIOR_binary_plane_coco_annotations_ALL.json

Here is the file structure used for these benchmark datasets

```
Benchmark Datasets
├── data
│   ├── dior
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages-test
│   │   └── JPEGImages-trainval
│   ├── rareplanes
│   │   └── real
|   │       ├── metadata_annotations
|   |       │   ├── instances_test_aircraft.json
|   |       │   └── instances_test_role.json
|   │       └── test
|   |           └── PS-RGB_tiled
│   ├── isaid
│   │   ├── annotations
│   │   |   ├── instances_val.json  # original - not used
|   │   │   └── val_binary_plane_coco_annotations_ALL.json # used for benchmarking
│   │   └── val
    └── nwpu_vhr10
        ├── ground truth
        └── positive image set
```

</details>


----

# Getting Started

This step we install all the necessary dependencies to train an object detector on a local machine. For this workshop we will be utilizing `detectron2` that runs on `PyTorch` as the framework for training DNNs. 
__Quick start__
* Step 1: clone workshop repo ([git required](https://git-scm.com/)):  `git clone https://github.com/naivelogic/OnlyPlanes.git`
* Step 2: navigate to repo folder:   `cd OnlyPlanes`
* Step 3: install dependencies with [Anaconda](https://www.continuum.io/downloads): `sh scripts/00_install_env.sh`
* Step 4: verify installation: `conda activate OP`

<details>
 <summary>Usage: Local vs AML</summary>
For this repository to simplify the usage, we will provide the code from a local computational perspective. In the paper, we utilized Azure Machine Learning for training and evaluating the performance of the models. Additionally, in the paper all data was stored on an Azure blob container. While Azure ML is great for scaling compute intensive workloads, as long as you meet the requirements below a single GPU can put utilized to reperform results.
</details>


### Requirements
* [Python](https://www.python.org/downloads/) > 3.5
* [Pytorch](http://pytorch.org/) > 1.9
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
* [Cuda](https://developer.nvidia.com/cuda-toolkit) > 11.0
* [Detectron2](https://github.com/facebookresearch/detectron2)
* _Make sure the CUDA, Pytorch and Detectron2 have the correct CUDA version if using GPU_

<details>
 <summary>special notes on getting started with requirements</summary>
My OS for this project was a ubuntu-18.04 Azure VM with a K8 GPU. I highly recommend using at least one GPU (w/ >20GB of memory) with the correct CUDA installed. Make sure the CUDA, Pytorch and Detectron2 have the correct CUDA version if using GPU. 
</details>


## Usage
------
This project contains various scripts to reproduce the results in the paper. 


### Dataset Preparations

Download and unzip the training and evaluation datasets either locally or in an Azure blob container. For each of the zip datasets, contains the coco annotations `.json` file and images in the `data` folder. Start by downloading the OnlyPlanes dataset using the following commands:

```
sudo chmod u+x scripts/01a_download_OnlyPlanes_dataset.sh
sh scripts/01a_download_OnlyPlanes_dataset.sh
```

<details>
 <summary>details on the structure of OnlyPlanes Dataset and annotations</summary>
  The structure of each dataset is as follows:

```
.
└── OnlyPlanes
    ├── LICENSE.txt
    ├── coco_annotations
    │   ├── all_labels_coco_annotations.json    # all plane labels including fine-grain attributes
    │   ├── binary_plane_coco_annotations.json  # use this for **training on binary planes** 1 class
    │   ├── role_coco_annotations.json          
    │   ├── civil_role_coco_annotations.json    
    │   ├── OnlyPlanes_Role_Counts.csv
    │   └── OnlyPlanes_metadata.csv
    └── images
        ├── 1.png
        └── ...
```
</details>



### Train model on Synthetic OnlyPlanes dataset


OnlyPlanes paper provides an object detection and a segmentation model. You can train the model by yourself or directly use the snapshot provided by us. The following script lets you train them using the configurations defined in the OnlyPlanes paper. 

To train the dataset run the following, also update the training config `.yaml` files as necessary (e.g., number of classes)
```sh
conda activate OP

TRAIN_IMG_DIR=/path/to/image/directory         #(e.g., datasets/OnlyPlanes/images)
TRAIN_COCO_JSON=/path/to/coco_annotations.json #(e.g., datasets/OnlyPlanes/coco_ds/OnlyPlanes_binary_plane_annotations.json)

VAL_IMG_DIR=/path/to/image/directory         
VAL_COCO_JSON=/path/to/coco_annotations.json 

OUTPUT_FOLDER=/path/to/output_training_results
TRAINCONFIG=/path/to/training_configs.yaml    #(e.g., configs/frcnn_r50_fpn_demo.yaml)
TRAIN_METHOD=bbox #bbox segm (rotated and obb not supported yet)
python train.py --output-folder $OUTPUT_FOLDER --config-file $TRAINCONFIG \
                --train_img_dir $TRAIN_IMG_DIR --train_coco_json $TRAIN_COCO_JSON \
                --val_img_dir $VAL_IMG_DIR --val_coco_json $VAL_COCO_JSON \
                --train_method $TRAIN_METHOD
```

### Evaluation on real images

The `inference.py` script has been provided to demo the performance of the trained detector or segmentation on a folder containing test images of real world aircrafts. 

```sh
INPUT_DIR=/path/to/images
OUTPUT_DIR=/path/to/save/predicted_images
CONF=0.5
USE_GPU=True
CKPT_PATH=/path/to/model/model_final.pth
NUM_IMAGES=2

python inference.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --conf ${CONF} \
                    --use_gpu ${USE_GPU} --ckpt_path ${CKPT_PATH} --num_images ${NUM_IMAGES}
```


## License 

The source code of this repository is released only for academic use. See the [license](LICENSE) file for details. 

## Notes

The codes of this repository are built upone the folloiwng open sources. Thanks to the authors for sharing the code!
* Pre-trained machine learning models and tools from [Detectron2](https://github.com/facebookresearch/detectron2)
* Synthetic dataset is generated by Microsoft Mixed Reality


----
## Citation

If you find this project useful or utilize the OnlyPlanes Dataset for your research, please kindly cite our paper.

```bibtex
@article{hale2022OnlyPlanes,
  title={OnlyPlanes: Incrementally Tuning Synthetic Training Datasets for Satellite Object Detection},
  author={Hale, Phillip and Tiday, Luke and Urbina, Pedro},
  number={arXiv:TBD},
  year={2022}
}
```