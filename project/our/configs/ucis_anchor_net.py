## ---------------------- DEFAULT_SETTING ----------------------
# Set the default scope for the configuration
default_scope = 'mmdet'
# Define custom imports for the configuration
custom_imports = dict(imports=['project.our.our_model'], allow_failed_imports=False)

# Base model configuration for SAM (Structured Attention Model) pre-trained model
sam_pretrain_name = "/xxxx/sam-vit-huge"
sam_pretrain_ckpt_path = "/xxxx/sam-vit-huge/pytorch_model.bin"


# Define default hooks that manage the behavior and properties of the training process
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Hook to measure time per iteration
    logger=dict(type='LoggerHook', interval=20),  # Logging hook that logs information every 20 iterations
    param_scheduler=dict(type='ParamSchedulerHook'),  # Hook to adjust learning parameters according to a schedule
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=1, save_best=['coco/bbox_mAP', 'coco/segm_mAP'],
                    rule='greater', save_last=True),  # Checkpointing to save model states
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Hook to set the seed for the sampler in distributed training
)

# Environment configuration specific to the compute hardware
env_cfg = dict(
    cudnn_benchmark=False,  # Whether to enable cudnn benchmarking for performance
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # Multiprocessing configuration
    dist_cfg=dict(backend='nccl')  # Distributed backend configuration
)

# Visualization backends for local visualization
vis_backends = [dict(type='LocalVisBackend')]
# Configuration for a local visualizer
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# Log processing configuration to manage how logs are processed
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Logging level for the system
log_level = 'INFO'
# Initial model weights to load (if any)
load_from = None
# Whether to resume training from the last checkpoint
# resume = True
resume = False

## ---------------------- MODEL ----------------------
# Size of the crop used in data preprocessing
crop_size = (1024, 1024)

# Augmentations applied to batches of data, specifically padding and masking configurations
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

# Preprocessor configuration for the dataset, including normalization and padding
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # Normalization mean
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],  # Normalization standard deviation
    bgr_to_rgb=True,  # Convert images from BGR to RGB
    pad_mask=True,  # Enable padding for masks
    pad_size_divisor=32,  # Ensure padding size is divisible by 32
    batch_augments=batch_augments  # Apply batch augmentations defined above
)

# Number of classes in the dataset
num_classes = 1
# Number of points per point set (specific to some segmentation tasks)
pointset_point_num = 5  # per pointset point

# Main model configuration
model = dict(
    type='UCISAnchor',  # Type of model
    data_preprocessor=data_preprocessor,  # Attach the data preprocessing configuration
    decoder_freeze=True,  # Freeze the decoder part of the model during training
    shared_image_embedding=dict(
        type='UCISSamPositionalEmbedding',
        hf_pretrain_name=sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        type='UCISSamVisionEncoder',
        hf_pretrain_name=sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path),
        peft_config=None,  # Parameter-efficient fine-tuning configuration (currently not used)
    ),
    adapter=dict(
        type='ChannelViTAdapters', adapter_layer=range(8, 33, 8), embed_dim=1280  # Configuration for the adapter layers
    ),
    neck=dict(
        type='UCISFPN',
        feature_aggregator=dict(
            type='UCISFeatureAggregator',
            in_channels=[1280] * (32 + 1),  # Input channels configuration
            out_channels=256,
            hidden_channels=64,
            select_layers=range(8, 33, 8),  # Layers to select for feature aggregation#
        ),
        feature_spliter=dict(
            type='UCISSimpleFPNHead',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='UCISPrompterAnchorRoIPromptHead',
        with_extra_pe=True,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='UCISPrompterAnchorMaskHead',
            mask_decoder=dict(
                type='UCISSamMaskDecoder',
                hf_pretrain_name=sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path)),
            in_channels=256,
            roi_feat_size=14,
            per_pointset_point=pointset_point_num,
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        )
    ),
    # Model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            #nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian', sigma=0.5),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=crop_size,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            #nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian', sigma=0.5),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            #nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian', sigma=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)
    )
)