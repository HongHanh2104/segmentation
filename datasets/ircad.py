import torch
from torch.utils import data
from torchvision import transforms
import pydicom as dicom

from utils.utils import NormMaxMin

import os
import glob

class IRCAD(data.Dataset):
    def __init__(self, root_path=None):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path 3D-IRCAD dataset!"
        self.root_path = root_path
        self.imgs_id = glob.glob(os.path.join(self.root_path, "3Dircadb1.[0-9]*/PATIENT_DICOM/image_[0-9]*"))
        
    def __getitem__(self, index):
        img_id = self.imgs_id[index]

        dicom_img = dicom.dcmread(img_id)
        arr_img = dicom_img.pixel_array

        arr_img_tf = transforms.Compose([
            transforms.ToTensor(),
            NormMaxMin()
        ])
        arr_img = arr_img_tf(arr_img)

        dicom_img = dicom.dcmread(img_id.replace('PATIENT_DICOM',
                                'MASKS_DICOM/liver'))
        mask_img = dicom_img.pixel_array
        mask_img_tf = transforms.Compose([
        ])
        mask_img = torch.Tensor(mask_img_tf(mask_img)).long() // 255
        
        return arr_img, mask_img

    def __len__(self):
        return len(self.imgs_id)
        
def test():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    dataset = IRCAD(root_path=args.root)

    for img, mask in dataset:
        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()
        plt.close()

if __name__ == "__main__":
    test()