angle_version = 'le90'
backend_args = None
custom_hooks = [
    dict(type='mmdet.SetEpochInfoHook'),
]
data_root = 'data/split_ss_dota/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
mask_filter_config = dict({
    0:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        required_metrics=[
            'aspect_ratio_reasonableness',
            'center_alignment',
            'color_consistency',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            color_consistency=2)),
    1:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        required_metrics=[
            'aspect_ratio_reasonableness',
            'center_alignment',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            color_consistency=2)),
    10:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        penalty_circularity=100,
        required_metrics=[
            'rectangularity',
            'circularity',
            'color_consistency',
            'aspect_ratio_reasonableness',
            'center_alignment',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            circularity=-3,
            color_consistency=2,
            rectangularity=6)),
    11:
    dict(
        required_metrics=[
            'circularity',
            'center_alignment',
            'color_consistency',
        ],
        weights=dict(center_alignment=10, circularity=5, color_consistency=2)),
    14:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        required_metrics=[
            'aspect_ratio_reasonableness',
            'center_alignment',
            'color_consistency',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            color_consistency=2)),
    2:
    dict(
        required_metrics=[
            'rectangularity',
            'color_consistency',
            'center_alignment',
        ],
        weights=dict(
            center_alignment=10, color_consistency=2, rectangularity=6)),
    3:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        penalty_circularity=100,
        required_metrics=[
            'rectangularity',
            'circularity',
            'color_consistency',
            'aspect_ratio_reasonableness',
            'center_alignment',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            circularity=-3,
            color_consistency=2,
            rectangularity=6)),
    7:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        required_metrics=[
            'rectangularity',
            'color_consistency',
            'aspect_ratio_reasonableness',
            'center_alignment',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            color_consistency=2,
            rectangularity=6)),
    8:
    dict(
        aspect_ratio_range=(
            1,
            5,
        ),
        penalty_circularity=100,
        required_metrics=[
            'rectangularity',
            'circularity',
            'color_consistency',
            'aspect_ratio_reasonableness',
            'center_alignment',
        ],
        weights=dict(
            aspect_ratio_reasonableness=5,
            center_alignment=10,
            circularity=-3,
            color_consistency=2,
            rectangularity=6)),
    9:
    dict(
        required_metrics=[
            'circularity',
            'center_alignment',
            'color_consistency',
        ],
        weights=dict(center_alignment=10, circularity=5, color_consistency=2)),
    'default':
    dict(
        required_metrics=[
            'color_consistency',
            'center_alignment',
        ],
        weights=dict(center_alignment=10, color_consistency=2))
})
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    bbox_head=dict(
        angle_coder=dict(
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0,
            type='PSCCoder'),
        edge_loss_cls=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            12,
            13,
        ],
        edge_loss_start_epoch=6,
        feat_channels=256,
        in_channels=256,
        joint_angle_start_epoch=1,
        loss_bbox=dict(loss_type='gwd', loss_weight=5.0, type='GDLoss'),
        loss_bbox_edg=dict(loss_weight=0.3, type='EdgeLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_overlap=dict(
            lamb=0, loss_weight=10.0, type='GaussianOverlapLoss'),
        loss_pgdm=dict(
            loss_weight=5.0,
            mask_filter_config=dict({
                0:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    required_metrics=[
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                        'color_consistency',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        color_consistency=2)),
                1:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    required_metrics=[
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        color_consistency=2)),
                10:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    penalty_circularity=100,
                    required_metrics=[
                        'rectangularity',
                        'circularity',
                        'color_consistency',
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        circularity=-3,
                        color_consistency=2,
                        rectangularity=6)),
                11:
                dict(
                    required_metrics=[
                        'circularity',
                        'center_alignment',
                        'color_consistency',
                    ],
                    weights=dict(
                        center_alignment=10,
                        circularity=5,
                        color_consistency=2)),
                14:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    required_metrics=[
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                        'color_consistency',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        color_consistency=2)),
                2:
                dict(
                    required_metrics=[
                        'rectangularity',
                        'color_consistency',
                        'center_alignment',
                    ],
                    weights=dict(
                        center_alignment=10,
                        color_consistency=2,
                        rectangularity=6)),
                3:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    penalty_circularity=100,
                    required_metrics=[
                        'rectangularity',
                        'circularity',
                        'color_consistency',
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        circularity=-3,
                        color_consistency=2,
                        rectangularity=6)),
                7:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    required_metrics=[
                        'rectangularity',
                        'color_consistency',
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        color_consistency=2,
                        rectangularity=6)),
                8:
                dict(
                    aspect_ratio_range=(
                        1,
                        5,
                    ),
                    penalty_circularity=100,
                    required_metrics=[
                        'rectangularity',
                        'circularity',
                        'color_consistency',
                        'aspect_ratio_reasonableness',
                        'center_alignment',
                    ],
                    weights=dict(
                        aspect_ratio_reasonableness=5,
                        center_alignment=10,
                        circularity=-3,
                        color_consistency=2,
                        rectangularity=6)),
                9:
                dict(
                    required_metrics=[
                        'circularity',
                        'center_alignment',
                        'color_consistency',
                    ],
                    weights=dict(
                        center_alignment=10,
                        circularity=5,
                        color_consistency=2)),
                'default':
                dict(
                    required_metrics=[
                        'color_consistency',
                        'center_alignment',
                    ],
                    weights=dict(center_alignment=10, color_consistency=2))
            }),
            sam_batch_size=4,
            sam_enable_predictor_cache=False,
            sam_instance_thr=4,
            sam_no_batch_inference=True,
            sam_sample_rules=dict(filter_pairs=[
                (
                    3,
                    10,
                    200,
                ),
            ]),
            type='PGDMLoss',
            use_class_specific_watershed=False),
        loss_ss=dict(loss_weight=1.0, type='Point2RBoxV2ConsistencyLoss'),
        num_classes=15,
        post_process=dict({11: 1.2}),
        square_cls=[
            1,
            9,
            11,
        ],
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='Point2RBoxV3Head',
        use_adaptive_scale=False,
        voronoi_thres=dict(
            default=[
                0.994,
                0.005,
            ],
            override=(
                (
                    [
                        2,
                        11,
                    ],
                    [
                        0.999,
                        0.6,
                    ],
                ),
                (
                    [
                        7,
                        8,
                        10,
                        14,
                    ],
                    [
                        0.95,
                        0.005,
                    ],
                ),
            )),
        voronoi_type='standard'),
    copy_paste_start_epoch=6,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    label_assign_pseudo_label_switch_eopch=6,
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='mmdet.FPN'),
    ss_prob=[
        0.68,
        0.07,
        0.25,
    ],
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.1, type='nms_rotated'),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=None,
    type='Point2RBoxV3')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=5e-05, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
sam_batch_size = 4
sam_enable_predictor_cache = False
sam_instance_thr = 4
sam_no_batch_inference = True
sam_sample_rules = dict(filter_pairs=[
    (
        3,
        10,
        200,
    ),
])
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='test/images/'),
        data_root='data/split_ss_dota/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/dota/Task1',
    type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=12)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        data_root='data/split_ss_dota/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                hbox_proportion=0,
                point_proportion=1.0,
                type='ConvertWeakSupervision'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        hbox_proportion=0, point_proportion=1.0,
        type='ConvertWeakSupervision'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
use_class_specific_watershed = False
use_sam = True
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        data_root='data/split_ss_dota/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'nocache_nobatch'
