# SLAB

## Dependenices
- torch
- torchvision
- numpy
- einops
- timm==0.4.12
- opencv-python==4.4.0.46
- termcolor==1.1.0
- yacs==0.1.8
- apex

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Train Models from Scratch

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <config-path> --data-path <imagenet-path> --output <output-path>
```

## Merge RepBN for SwinTransformer

```shell
python -m torch.distributed.launch --nproc_per_node=1 eval.py --cfg <config-path> --batch-size 128 --data-path <imagenet-path>  --pretrained <pretrained-path>
```
