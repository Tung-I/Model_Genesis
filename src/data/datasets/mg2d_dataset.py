import csv
import glob
import torch
import numpy as np
import nibabel as nib
import copy
import SimpleITK as sitk

from src.data.datasets.base_dataset import BaseDataset
# from src.data.transforms import compose
from src.data.transforms import Compose, ToTensor


class MG2DDataset(BaseDataset):
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
        self.train_preprocessings = Compose.compose(train_preprocessings)
        self.valid_preprocessings = Compose.compose(valid_preprocessings)
        self.transforms = Compose.compose(transforms)
        self.to_tensor = ToTensor()
        self.augments = Compose.compose(augments)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.


        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for _path, split_type in rows:
                if split_type == type_:
                    self.data_paths.append(_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        image = np.load(image_path)

        hu_max = 1000
        hu_min = -1000
        image = (image - hu_min) / (hu_max - hu_min)
        image[image>1] = 1.
        image[image<0] = 0.
        # image = np.expand_dims(image, 2)  # (H, W) -> (H, W, C=1)
        image = image[..., None]
        # image = np.stack([image, image, image], 2)  # (H, W, D, C)
        # image = image[..., None, None]
        label = copy.deepcopy(image)

        # normalize and crop
        if self.type == 'train':
            # image, label = self.train_preprocessings(image, label, normalize_tags=[True, True], target=label, target_label=2)
            image, label = self.train_preprocessings(image, label, normalize_tags=[True, True])
            # image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            # image, label = self.valid_preprocessings(image, label, normalize_tags=[True, True], target=label, target_label=2)
            image, label = self.valid_preprocessings(image, label, normalize_tags=[True, True])
        # transformations
        image, = self.transforms(image)

        # image = image[:, :, 0, 0:1]
        # label = label[:, :, 0, 0:1]
        image, label = self.to_tensor(image, label, dtypes=[torch.float, torch.float])
        # image, label = image.permute(2, 0, 1).contiguous(), label.permute(2, 0, 1).contiguous()



        return {"image": image, "label": label}
