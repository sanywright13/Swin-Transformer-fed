# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
from imblearn.over_sampling import SMOTE
import torch
from collections import Counter
from .zipreader import is_zip_path, ZipReader
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

    return images
import numpy as np
# Create a new dataset with SMOTE-applied data
def makeBreastnistdata(root_path, prefix):
  data_path=os.path.join(root_path,'feddata')
  #print(f'data source {data_path}')
  medmnist_data=os.path.join(data_path,'breastmnist.npz')
  data=np.load(medmnist_data)
  np.load(medmnist_data)
  #test_data=data['test_images']
  #test_label=data['test_labels']
  print(prefix)
  if prefix=='train':
    train_data=data['train_images']
    train_label=data['train_labels']
    #print(f'train_data shape:{train_data[0]}')

    return train_data , train_label
  else:
    val_data=data['val_images']
    val_label=data['val_labels']
    #print( f'valid data shape {val_data.shape}')
    return val_data , val_label


class BreastMnistDataset(data.Dataset):
    
    def __init__(self,root,prefix, transform=None):
      data,labels= makeBreastnistdata(root, prefix)
      self.data=data
      self.labels  = labels  
      if prefix=='train':
        print(prefix)
        #num_samples, *image_shape = data.shape
        #flattened_data = data.reshape(num_samples, -1)
        # Apply SMOTE to balance the classes
        #adasyn = ADASYN(sampling_strategy='minority')

        #resampled_data, resampled_labels = adasyn.fit_resample(flattened_data, labels) 
        # Reshape the data back to original image shape
        #self.data = resampled_data.reshape(-1, *image_shape)
        #print(f'sanaa data shape {self.data.shape}')
        #self.labels = resampled_labels
        # Print resampled class distribution
        #resampled_counts = Counter(resampled_labels)
      self.transform = transform
    def __len__(self):
        self.filelength = len(self.labels)
        return self.filelength

    def __getitem__(self, idx):
        #print(f'data : {self.data[idx]}')
        image =self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        #print(f"Resampled {image.shape}")

        
        return image, label
class SMOTEDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no"):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
      assert self.cache_mode in ["part", "full"]
      n_sample = len(self.samples)
      global_rank = 0  # Default to single rank for CPU
      world_size = 1   # Single process for CPU training

      samples_bytes = [None for _ in range(n_sample)]
      start_time = time.time()
      for index in range(n_sample):
        if index % (n_sample // 10) == 0:
            t = time.time() - start_time
            print(f'cached {index}/{n_sample} takes {t:.2f}s per block')
            start_time = time.time()
        path, target = self.samples[index]
        if self.cache_mode == "full":
            samples_bytes[index] = (ZipReader.read(path), target)
        elif self.cache_mode == "part" and index % world_size == global_rank:
            samples_bytes[index] = (ZipReader.read(path), target)
        else:
            samples_bytes[index] = (path, target)
      self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            #return img.convert('RGB')
    #return img.convert('RGB')
    # Convert image mode if necessary
    if img.mode != 'L':  # Keep grayscale if already in 'L'
        img = img.convert('L')
    return img


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CachedImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples
        # Print the shape and number of channels
        #print(f"Transformed image shape: {self.imgs[0]}")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        # It should be 'L' for grayscale
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Print the shape and number of channels
        #print(f"Transformed image shape: {img.shape}")
        #print(f"Number of channels: {img.shape[0]}")  # img.shape[0] is the number of channels
        print(f"Original image value: {img}")
        return img, target
