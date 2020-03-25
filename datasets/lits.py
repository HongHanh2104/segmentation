import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

import nibabel as nib
import pydicom as pdc
import SimpleITK as sitk

import os
import glob
from pathlib import Path
import re

from utils.utils import NormMaxMin


class LiTSSingle(data.Dataset):
    def __init__(self, root_path=None):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path LiTS dataset!"
        self.root_path = Path(root_path)

        self.imgs_id = self.root_path.rglob('PATIENT/*.dcm')

        regex = re.compile(
            r'.*LiTS(?P<patient_id>[0-9]+).*/(?P<slice_id>[0-9]+).dcm')

        def get_sort_key(x):
            match = regex.match(str(x))
            patient_id = int(match.group('patient_id'))
            slice_id = int(match.group('slice_id'))
            return (patient_id, slice_id)

        self.imgs_id = sorted(self.imgs_id, key=get_sort_key)
        self.imgs_id = list(map(str, self.imgs_id))

    def __getitem__(self, index):
        img_id = self.imgs_id[index]

        nii_img = pdc.dcmread(img_id)
        arr_img = nii_img.pixel_array
        arr_img_tf = transforms.Compose([
            NormMaxMin()
        ])
        arr_img = arr_img_tf(torch.Tensor(arr_img)).unsqueeze(0)

        nii_img = pdc.dcmread(img_id.replace('PATIENT', 'LABEL'))
        label_img = nii_img.pixel_array
        label_img_tf = transforms.Compose([
        ])
        label_img = torch.Tensor(label_img_tf(label_img)).long()

        return arr_img, label_img

    def __len__(self):
        return len(self.imgs_id)


def test():
    import torchvision as tv
    import matplotlib.pyplot as plt
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    dataset = LiTSSingle(root_path=args.root)

    now = time.time()
    for img, mask in dataset:
        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.tight_layout()
        plt.show()
        plt.close()
    print(time.time() - now)


if __name__ == "__main__":
    test()
