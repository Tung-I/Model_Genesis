import csv
import glob
import torch
import numpy as np
import nibabel as nib
import copy
import SimpleITK as sitk

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class MGDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the 3D segmentation method.

    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, to_tensor, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.to_tensor = compose(to_tensor)
        self.augments = compose(augments)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for dir_name, split_type in rows:
                if split_type == type_:
                    paths = sorted(list((self.data_dir / dir_name).glob('*.mhd')))
                    self.data_paths.extend(paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        itkimage = sitk.ReadImage(str(image_path))
        image = sitk.GetArrayFromImage(itkimage)
        label = copy.deepcopy(image)

        # (D, H, W) -> (H, W, D, C)
        image, label = image.transpose(1, 2, 0)[..., None], label.transpose(1, 2, 0)[..., None]

        # normalize and crop
        if self.type == 'train':
            image, label = self.train_preprocessings(image, label, normalize_tags=[True, True], target=label, target_label=2)
            image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            image, label = self.valid_preprocessings(image, label, normalize_tags=[True, True], target=label, target_label=2)
        # transformations
        image = self.transforms(image, dtypes=[torch.float])

        image, label = self.to_tensor(image, label)
        # (H, W, D, C) -> (C, D, H, W)
        image, label = image.permute(3, 2, 0, 1).contiguous(), label.permute(3, 2, 0, 1).contiguous()



        return {"image": image, "label": label}
