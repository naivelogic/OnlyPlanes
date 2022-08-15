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