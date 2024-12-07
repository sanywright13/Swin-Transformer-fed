# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from imblearn.over_sampling import SMOTE

from timm.data import create_transform
from torch.utils.data import DataLoader, RandomSampler
from .cached_image_folder import CachedImageFolder , SMOTEDataset
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp



def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.MODEL.NUM_CLASSES=2
    # Get a sample from the dataset (assuming dataset_train is an instance of a Dataset class)
    sample, label = dataset_train[0]  # Access the first sample and label

    # Print the shape and number of channels
    #print(f"Sample shape: {sample.shape}")  # Shape of the image tensor
    #print(f"Number of channels: {sample.shape[0]}")  # Channels are represented by the first dimension (C, H, W)

    config.freeze()
    #global_rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    # Assuming dataset_train is built using your existing code
    # Extract data from dataset_train
    features = []
    labels = []
    for i in range(len(dataset_train)):
      img, label = dataset_train[i]
      features.append(img.flatten().numpy())  # Flatten image tensor to 1D array
      labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # Reshape features back to original image dimensions
    features_resampled = features_resampled.reshape(-1, *dataset_train[0][0].shape)

    # Convert back to PyTorch tensors
    features_resampled = torch.tensor(features_resampled).float()
    labels_resampled = torch.tensor(labels_resampled).long()


    # Build the resampled dataset
    resampled_dataset_train = SMOTEDataset(features_resampled, labels_resampled)
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    #num_tasks = dist.get_world_size()
    #global_rank = dist.get_rank()
  

    # Remove distributed training setup and use default sampler
    #sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        resampled_dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,shuffle=True
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    #print(f'gggg {transform}')
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    #print(f'resize_im : {resize_im}')
    #if is_train:
        # this should always dispatch to transforms_imagenet_train
    '''
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
  '''
    t = []
        #print(f"Transformed image shape: {transform.shape}")

        #if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            #transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )
    t.append(transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4))
    t.append(transforms.Grayscale(num_output_channels=1))  # Keep single channel
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    t.append(transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)))
    t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    t.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    t.append(t.append(transforms.ToTensor()))
    # Check dimensionality after transformation
    def check_dim(img):
        #print(f"Image shape after transform: {img.shape}")  # Print shape
        return img
    
    t.append(check_dim)
    #t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t.append(transforms.Normalize([0.5], [0.5]))  # For grayscale data
    return transforms.Compose(t)
