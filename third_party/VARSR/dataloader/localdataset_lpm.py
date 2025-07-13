import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torch.utils import data as data
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from .realesrgan import RealESRGAN_degradation
from myutils.img_util import convert_image_to_fn
import pickle
from transformers import AutoProcessor
import torch.nn.functional as F
from test_varsr import pt_to_numpy, numpy_to_pil
import pandas as pd
import scipy.io
def exists(x):
    return x is not None


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


class LocalImageDataset_LPM(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets/6393/3658/datasets/sr_datasets/train_pasd_datasets/pngtxt_dir/", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(LocalImageDataset_LPM, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio

        self.degradation = RealESRGAN_degradation('/home/quyunpeng/hart/dataloader/params_realesrgan.yml', device='cpu')
        self.resize_scale = 1.25
        center_crop = True

        self.crop_preproc = transforms.Compose([
            transforms.Resize(round(self.resize_scale*image_size), interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.neg_resize_preproc = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.toPIL = transforms.ToPILImage()
        
        self.img_paths = []
        self.neg_paths = []
        pngtxt_dir = "trainset/"
        data_folders = os.listdir(pngtxt_dir)
        for data_folder in data_folders:
            self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{data_folder}/*.png'))[:])
        pngtxt_dir = "trainset_neg/"
        data_folders = os.listdir(pngtxt_dir)
        for data_folder in data_folders:
            self.neg_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{data_folder}/*.png'))[:])


        self.labels = torch.zeros(len(self.img_paths))
        self.neg_labels = torch.ones(len(self.neg_paths))
        self.img_labels = torch.cat((self.labels, self.neg_labels), dim=0).tolist()
        self.img_paths.extend(self.neg_paths)
        print(len(self.img_paths))


    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        label_B = self.img_labels[index]
        txt_path = img_path.replace(".png", ".txt")
        image = Image.open(img_path).convert('RGB')

        if label_B == 0:
            image = self.crop_preproc(image)
        elif label_B==1:
            image = self.neg_resize_preproc(image)
        GT_image_t, LR_image_t = self.degradation.degrade_process(np.asarray(image)/255., resize_bak=self.resize_bak)
        example["conditioning_pixel_values"] = LR_image_t.squeeze(0) * 2.0 - 1.0
        example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0
        example["label_B"] = int(label_B)
        example['img_path'] = img_path

        return example

    def __len__(self):
        return len(self.img_paths)