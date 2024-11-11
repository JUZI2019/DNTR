# dataset settings
dataset_type = 'Sen1shipDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(608, 608), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
        # transforms=[
        #     dict(type='RResize'),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='Pad', size_divisor=32),
        #     dict(type='DefaultFormatBundle'),
        #     dict(type='Collect', keys=['img'])
        # ]
        )
]

data_root = '/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file = data_root + 'train/sen1ship_train.json',
        img_prefix = data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file = data_root + 'test/sen1ship_test.json',
        img_prefix = data_root + 'test/images/',
        pipeline = test_pipeline),
    test=dict(
        type = dataset_type,
        ann_file = data_root + 'test/sen1ship_test.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline)
    # test=dict(
    #     type = dataset_type,
    #     ann_file = data_root + 'offshore_test/sen1ship_offshore_test.json',
    #     img_prefix=data_root + 'offshore_test/images/',
    #     pipeline=test_pipeline)        

    # test=dict(
    #     type = dataset_type,
    #     ann_file = data_root + 'inshore_test/sen1ship_inshore_test.json',
    #     img_prefix=data_root + 'inshore_test/images/',
    #     pipeline=test_pipeline)        

    )

