# model settings
file_name = 'ma800_mi50_b6_t99_1.pkl'
# file_name = 'uni_50.pkl'
rs = 'mix_mixup'
model = dict(
    type='SimpleClassifier',
    pretrained='torchvision://resnet50',
    lock_back=False,
    lock_neck=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_classes=8,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0.5),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=8,
        method='fc',
        lock_cls=False,
        loss_cls=dict(
            type='PGLoss', 
            use_sigmoid=True,
            reweight_func='rebalance',
            # reweight_func='None',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0, 
            freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq1.pkl',
            class_split='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_split.pkl',
            )))
        # loss_cls=dict(
        #     type='MultiCELoss',
        #     gamma=1,m=0,bs=False,
        #     freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq1.pkl',
        #     use_sigmoid=True,
        #     reweight_func='rebalance',
        #     # reweight_func='None',
        #     focal=dict(focal=False, balance_param=2.0, gamma=2),
        #     logit_reg=dict(init_bias=0.05, neg_scale=5),
        #     map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
        #     loss_weight=1.0, num_classes = 8,
        #     # freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq.pkl',
        #     )))
        # loss_cls=dict(
        #     type='BSExpertLoss',
        #     freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq1.pkl',
        #     )))
        # loss_cls=dict(
        #     type='SeesawLoss',
        #     p=0.8, q=2, num_labels=8,
        #     )))
        # loss_cls=dict(
        #     type='EQLv2',
        #     num_classes=8,
        #     )))
        # loss_cls=dict(
        #     type='MultiCELoss',
        #     freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq.pkl')))
        # loss_cls=dict(
        #     type ='AsymmetricLoss', gamma_neg=4,gamma_pos=1,n_clip=0.5,p_clip=0.0)))
        # loss_cls=dict(
        #     type='ResampleLoss', 
        #     use_sigmoid=True,
        #     reweight_func='rebalance',
        #     # reweight_func='None',
        #     focal=dict(focal=True, balance_param=2.0, gamma=2),
        #     logit_reg=dict(init_bias=0.05, neg_scale=2),
        #     map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
        #     loss_weight=1.0, num_classes = 8,
        #     freq_file='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_freq1.pkl',
        #     class_split='D:/dataset/semanticSeg/Semantic_Dataset/Code/DDRNet.pytorch-main/class_split.pkl',
        #     )))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'D:/dataset/voc/'
online_data_root = 'appendix/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler = 'random',
    # sampler='ClassAware',
    sampler_cfg = dict(
        reduce = 4,
    ),
    train=dict(
            type=dataset_type,
            # ann_file=online_data_root + 'longtail2012/img_id_' + file_name,
            ann_file=online_data_root + 'longtail2012/img_id.txt',
            img_prefix=data_root + 'VOC2012/',
            img_scale=(224, 224),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0,
        val_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        class_split=online_data_root + 'longtail2012/class_split.pkl',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam',lr = 3e-5)
# optimizer_config = dict()
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 3,
    step=[5])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ]) 
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/' + file_name[:-4]
# work_dir = './work_dirs/LT_voc_resnet50_pfc_DB'
work_dir = './work_dirs/LT_voc_resnet50_pfc_DB'
load_from = None
resume_from = None
workflow = [('train', 1)]
