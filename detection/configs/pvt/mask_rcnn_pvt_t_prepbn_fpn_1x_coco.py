_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # pretrained='pretrained/pvt_tiny.pth',
    backbone=dict(
        type='pvt_tiny',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))
optimizer_config = dict(grad_clip=None)


# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )