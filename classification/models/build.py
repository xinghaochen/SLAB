# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .slab_swin import SlabSwinTransformer
from .slab_pvt import slab_pvt_tiny, slab_pvt_small, slab_pvt_medium, slab_pvt_large
from .slab_cswin import SLAB_CSWin_64_24181_tiny_224, SLAB_CSWin_64_24322_small_224
from .slab_pvt_v2 import slab_pvt_v2_b0, slab_pvt_v2_b1, slab_pvt_v2_b2, slab_pvt_v2_b3, \
    slab_pvt_v2_b4, slab_pvt_v2_b5
from .swin_transformer import SwinTransformer
from .slab_deit import deit_tiny_patch16_224_linear_repbn, deit_small_patch16_224_linear_repbn
from .deit import deit_tiny_patch16_224_repbn, deit_small_patch16_224_repbn
from timm.models import create_model


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type in ['SwinTransformer']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'patch_size=config.MODEL.SWIN.PATCH_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'embed_dim=config.MODEL.SWIN.EMBED_DIM,'
                                  'depths=config.MODEL.SWIN.DEPTHS,'
                                  'num_heads=config.MODEL.SWIN.NUM_HEADS,'
                                  'window_size=config.MODEL.SWIN.WINDOW_SIZE,'
                                  'mlp_ratio=config.MODEL.SWIN.MLP_RATIO,'
                                  'qkv_bias=config.MODEL.SWIN.QKV_BIAS,'
                                  'qk_scale=config.MODEL.SWIN.QK_SCALE,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'ape=config.MODEL.SWIN.APE,'
                                  'patch_norm=config.MODEL.SWIN.PATCH_NORM,'
                                  'use_checkpoint=config.TRAIN.USE_CHECKPOINT,'
                                  'fused_window_process=config.FUSED_WINDOW_PROCESS)')
    elif model_type in ['SlabSwinTransformer']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'patch_size=config.MODEL.SWIN.PATCH_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'embed_dim=config.MODEL.SWIN.EMBED_DIM,'
                                  'depths=config.MODEL.SWIN.DEPTHS,'
                                  'num_heads=config.MODEL.SWIN.NUM_HEADS,'
                                  'window_size=config.MODEL.SWIN.WINDOW_SIZE,'
                                  'mlp_ratio=config.MODEL.SWIN.MLP_RATIO,'
                                  'qkv_bias=config.MODEL.SWIN.QKV_BIAS,'
                                  'qk_scale=config.MODEL.SWIN.QK_SCALE,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'ape=config.MODEL.SWIN.APE,'
                                  'patch_norm=config.MODEL.SWIN.PATCH_NORM,'
                                  'use_checkpoint=config.TRAIN.USE_CHECKPOINT,'
                                  'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                  'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                  'attn_type=config.MODEL.LA.ATTN_TYPE)')
    elif model_type == 'deits':
        model = create_model(
            config.MODEL.NAME,
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_block_rate=None,
        )
    elif model_type in ['SLAB_CSWin_64_24181_tiny_224', 'SLAB_CSWin_64_24322_small_224']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                  'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                  'attn_type=config.MODEL.LA.ATTN_TYPE,'
                                  'la_split_size=config.MODEL.LA.CSWIN_LA_SPLIT_SIZE)')

    elif model_type in ['slab_pvt_tiny', 'slab_pvt_small', 'slab_pvt_medium', 'slab_pvt_large',
                        'slab_pvt_v2_b0', 'slab_pvt_v2_b1', 'slab_pvt_v2_b2',
                        'slab_pvt_v2_b3', 'slab_pvt_v2_b4', 'slab_pvt_v2_b5']:
        model = eval(model_type + '(drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                  'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                  'attn_type=config.MODEL.LA.ATTN_TYPE,'
                                  'la_sr_ratios=str(config.MODEL.LA.PVT_LA_SR_RATIOS))')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
