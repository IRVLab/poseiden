# Poseiden

This is the official code release for our paper "Stereo-Based 3D Human Pose Estimation for Underwater Robots Without 3D Supervision". **[[Paper](https://ieeexplore.ieee.org/document/10947328)]  [[Project Page](https://human-pose-underwater-3d.github.io)]**

<img src="images/demo_mads.gif" alt="demo_mads" width="600"/>

**Poseiden** (**Pose** **I**n **D**ynamic **En**vironment) is a stereo-based 3D human pose estimation model capable of providing **absolute-scale 3D human poses** from stereo image pairs. Surpass the limitation of dynamic environments like underwater where 3D ground truths are extermely challenging to acquire, the model only requires 2D groud truths for training.

This repository is the implementation of the stereo-based 3D human pose estimation model proposed in the paper. For more information about the auto-refinement pipeline preposed in the paper, please refer to [DiverPose-AutoRefinement]() (Coming Soon). Note that the model in this repository has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

## Table of Contents

- [Environment Setup](#EnvironmentSetup)
- [Datasets](#Datasets)
- [Train](#Train)
- [Test](#Test)

<a name="EnvironmentSetup"></a>
## Environment Setup

1. Build Docker Image
    ```
    docker build -t diverpose docker/
    ```
    **All codes within this repository should be able to run under this docker environment.**

2. Run Docker container:
    * Change the ```$WORKSPACE_DIR``` variables in [run_container.sh](run_container.sh) to the path where you store this repository before running the following command.
    ```
    bash run_container.sh
    ```

<a name="Datasets"></a>
## Datasets

### COCO (For Model Pretrain)

* Poseiden requires pretraining process to enhance feature representations in transformer layers.
1. Download the 2017 train and val images and annotations from [COCO Keypoints Dataset](https://cocodataset.org/#download).
2. Move the data folder into ```data/``` and structure as follows:
    ```
    data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
    ```

### MADS

1. Download MADS_depth and MADS_multiview from [MADS: Martial Arts, Dancing, and Sports Dataset](http://visal.cs.cityu.edu.hk/research/mads/) 
2. Run ```extract_mads_data.py``` to extract images from videos
    ```bash
    python extract_mads_data.py \
        --depth_data_path <PATH_TO_MADS_depth> \
        --multiview_data_path <PAth_TO_MADS_multiview> \
        --output_path data/MADS_extract \
        --rectify
    ```
    * Note: the root value in ```conf/dataset/mads.yaml``` should be the same as the directory set for output_path (```data/MADS_extract``` by default)
3. (Optional) Visualize the data to check if loaded correctly
    ```
    python helpers/display_data_3d.py --config-name train_stereo dataset=mads
    ```
    
### DiverPose

* One of the key contributions of this paper is to automatically refine human annotations from stereo keypoints. Please refer to [DiverPose-AutoRefinement]() for more details and guideline for download.
* Once extracts image and annotations, put the data folder under ```data/```.
* (Optional) Visualize the data to check if loaded correctly
    ```
    python helpers/display_data_3d.py --config-name train_stereo dataset=diver
    ```
<a name="Train"></a>
## Train

We use the Hydra library to manage configurations. For more information, please refer to the [Hydra documentation](https://hydra.cc/docs/intro/).

### Pretrain
```
python train.py --config-name train_mono name=<CUSTOM_NAME_FOR_MODEL> dataset=coco
```

### MADS
```
python train.py \
    --config-name train_stereo \
    name=name=<CUSTOM_NAME_FOR_MODEL> \
    model.backbone=gelanS \
    dataset=mads \
    model.pretrained=<PATH_TO_PRETRAIN_MODEL_WEIGHT> \
    model.dmin=5 \
    model.dmax=30 \
```

### DiverPose
```
python train.py \
    --config-name train_stereo \
    name=name=<CUSTOM_NAME_FOR_MODEL> \
    model.backbone=gelanS \
    dataset=diver \
    model.pretrained=<PATH_TO_PRETRAIN_MODEL_WEIGHT> \
    model.dmin=2 \
    model.dmax=15 \
```

* Note that you can also set ```model.pretrained=""``` to avoid loading weights from pretrained model.

<a name="Test"></a>
## Test
**Make sure the mode configuration (dmin, dmax, backbone, etc.) used for testing are same for training.**

### MADS
```
python test.py \
    --config-name test_stereo \
    dataset=mads \
    model.backbone=gelanS \
    model.dmin=5 \
    model.dmax=30 \
    model_weight=<PATH_TO_MODEL_WEIGHT> \
    visualize=False (if true, visualize the estimations and ground truths)
```
* Note: model weight is also provided [here](https://drive.google.com/drive/folders/1R5QEbIws7daK7pu5d98ZpElrbevbjUWe?usp=sharing) for demo

### DiverPose

* Due to the difficulity in collecting 3D ground truths underwater, we collect pseudo ground truths data for validation instead. Please refer to the paper for more details.
* Download [test]() data (coming soon)
* Run:
    ```
    python test_diver.py \
        --config-name test_diver \
        model.backbone=gelanS \
        model.dmin=2 \
        model.dmax=15 \
        data_path=<PATH_TO_TEST_DATA> \
        model_weight=<PATH_TO_MODEL_WEIGHT> \
        yolo_weight=<PATH_TO_ONNX_MODEL>
    ```
* Note: model weight is also provided [here](https://drive.google.com/drive/folders/1nCwcSAPVvDDZbV4ItBoK5X6MvPaQ-cxH?usp=sharing) for demo
* The YOLOv8 onnx model is used here to locate the diver and crop the region of it from the entire image. Please refer to [DiverPose-AutoRefinement]() for more details or download the model weight [here](https://drive.google.com/file/d/1nc-is6bRzSHgARuEppfmTGcdE6hX_N8q/view?usp=share_link) for convenience.


## Acknowledgments

Several functions in this repository are adapted and modified from [TransPose](https://github.com/yangsenius/TransPose) and [mmpose](https://github.com/open-mmlab/mmpose).

## Citation

If you use this code or the DiverPose dataset for your research, please cite:
```
@ARTICLE{10947328,
  author={Wu, Ying-Kun and Sattar, Junaed},
  journal={IEEE Robotics and Automation Letters}, 
  title={Stereo-Based 3D Human Pose Estimation for Underwater Robots Without 3D Supervision}, 
  year={2025},
  pages={1-8},
  doi={10.1109/LRA.2025.3557235}}
```
