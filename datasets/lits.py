import torch
from torch.utils import data
from torchvision import transforms

import nibabel as nib
import SimpleITK as sitk
import os
import glob
from pathlib import Path
from utils.utils import NormMaxMin

class LiTS(data.Dataset):
    def __init__(self, root_path=None):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path LiTS dataset!"
        self.root_path = Path(root_path)
        self.imgs_id = [x for x in sorted(glob.glob(str(self.root_path / 'volume-*')), 
                            key=lambda x: int(x.split("-")[-1].split(".")[0]))]

    def __getitem__(self, index):
        img_id = self.imgs_id[index]
        
        nii_img = nib.load(img_id)
        arr_img = nii_img.get_fdata()
        arr_img_tf = transforms.Compose([
            NormMaxMin()
        ])
        arr_img = arr_img_tf(torch.Tensor(arr_img)).permute(2, 1, 0) \
                                                   .unsqueeze(1)
        
        nii_img = nib.load(img_id.replace('volume', 'segmentation'))
        label_img = nii_img.get_fdata()
        label_img_tf = transforms.Compose([
            
        ])
        label_img = torch.Tensor(label_img_tf(label_img))
        
        return arr_img, label_img
        

def test():
    import torchvision as tv
    import matplotlib.pyplot as plt
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args() 

    dataset = LiTS(root_path=args.root)

    now = time.time()
    for img, label in dataset:
        print(img.shape)
        # slices = tv.utils.make_grid(img, nrow=15)
        # plt.imshow(slices.permute(1, 2, 0))
        # plt.tight_layout()
        # plt.show()
        # plt.close()
    print(time.time() - now)

    now = time.time()
    for img, label in dataset:
        print(img.shape)
    print(time.time() - now)

if __name__ == "__main__":
    test()
