# Robust Tracking Module
Implementation of the Robust Tracking Module on SiamRPN

![RTM Module](Assets/RTM_diag.png)

## Pretrained model

[GDrive](https://drive.google.com/file/d/1TbQztkEbEp4wRAZl-z18U4lNxYn9gSp9/)

## Setup environment

```
conda env create -f environment.yaml
conda activate rtm
```

## Training

coming soon

## Run tracking

### Dataset evaluation
- Set correct paths for each dataset in ```configs/local.py```
- Download pretrained model and save in ```checkpoints/model_RTM.pth```
- ```python tracking_evaluation.py --dataset lasot --distortion original```

### Run demo
- Run on video: ```python demo.py path/to/video.mp4```
- Run on webcam: ```python demo.py 0```
- Distort input with ```--distortion``` argument:
    - White Gaussian Noise: ```WGN```
    - Salt and Pepper: ```SnP```
    - Gaussian Blur: ```GB```
- Select initial bounding box with mouse (click and drag)


## Related repositories
- [DaSiamRPN](https://github.com/foolwood/DaSiamRPN)
- [Siamese-RPN-pytorch](https://github.com/songdejia/Siamese-RPN-pytorch)
- [SiamRPN Pytorch](https://github.com/huanglianghua/siamrpn-pytorch)
- [GOT-10k toolkit](https://github.com/got-10k/toolkit)
