# Transformer with RepBN for Object Detection

## Installation

**Step 1** Install pytorch

```shell
pip install torch 
pip install torchvision
```

**Step 2** Install packages

```shell
pip install timm==0.4.12
pip install einops
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
pip install -U openmim
pip install mmcv-full==1.4.0
pip install mmdet==2.11.0
```

**Step 3** Install apex

## Dataset
The coco dataset should be prepared as follows:

```
$ tree data
data
└── coco2017
    ├── train2017
    ├── val2017
    └── annotations
```

## Train Model

SLAB-Swin-T
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/swin/mask_rcnn_swin_linear_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```

SLAB-Swin-S
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/swin/mask_rcnn_swin_linear_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```

Swin-T-RepBN
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```

Swin-S-RepBN
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```

PVT-T-RepBN
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/pvt/mask_rcnn_pvt_t_fpn_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```

PVT-S-RepBN
```shell
python -m torch.distributed.launch --nproc_per_node 8 --nnodes <world_size> --node_rank <rank> train.py configs/pvt/mask_rcnn_pvt_s_fpn_1x_coco.py --work-dir <output_path> --launcher pytorch --init_method <init_method> --cfg-options model.pretrained=<pretrained_backbone_path>
```
