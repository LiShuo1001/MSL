# MSL
The code of Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection


## Features

Please get the VideoSwin features and labels on the ShanghaiTech and UCF-Crime datasets from here: [MSL-VideoSwin-Feature](https://drive.google.com/drive/folders/12G8HSj7sFK60QHHnGFriFqL4Ld94YgUy?usp=sharing)。

* The checkpoint of VideoSwin we used is `swin_base_patch244_window877_kinetics400_1k.pth`, you can download from [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) or [github pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth). 

* The configuration file used to extract the features of the ShanghaiTech dataset:
```
_base_ = ['../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py']
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.3), test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'MyDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=(416, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1, workers_per_gpu=1,
    test=dict(type=dataset_type, ann_file=None, data_prefix=None, pipeline=test_pipeline)
)
```

* The configuration file used to extract the features of the UCF-Crime dataset:
```
_base_ = ['../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py']
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.3), test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'MyDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=(288, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1, workers_per_gpu=1,
    test=dict(type=dataset_type, ann_file=None, data_prefix=None, pipeline=test_pipeline)
)
```


## TODO

The code will be released, please wait.

