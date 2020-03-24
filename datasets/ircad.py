import torch
from torch.utils import data
from torchvision import transforms as tvtf
import pydicom as dicom

from utils.utils import NormMaxMin
from transforms.crop import RandomCrop

import os
import glob
from pathlib import Path
import re

class IRCADSeries(data.Dataset):
    def __init__(self, root_path=None, is_train=True):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path 3D-IRCAD dataset!"
        self.root_path = Path(root_path)
        self.series_id = []
        for folder_name in os.listdir(self.root_path):
            img_lst = [ str(self.root_path / folder_name / "PATIENT_DICOM" / x) for x in sorted(
                            os.listdir(self.root_path / folder_name / "PATIENT_DICOM"),
                            key=lambda x: int(x.split('_')[-1]))]
            self.series_id.append(img_lst)
        self.is_train = is_train
    
    def __getitem__(self, index):
        serie_id = self.series_id[index]
        dicom_serie = [dicom.dcmread(x) for x in serie_id]
        arr_serie = [x.pixel_array for x in dicom_serie]
        arr_serie_tf = tvtf.Compose([
            tvtf.ToTensor(),
            NormMaxMin()
        ])
        arr_serie = [arr_serie_tf(x).unsqueeze(0) for x in arr_serie]
        arr_serie = torch.cat(arr_serie)
        
        dicom_serie = [dicom.dcmread(x.replace('PATIENT_DICOM',
                        'MASKS_DICOM/liver')) for x in serie_id]
        mask_serie = [x.pixel_array for x in dicom_serie]
        mask_serie_tf = tvtf.Compose([
        ])
        mask_serie = [torch.Tensor(mask_serie_tf(x)).long().unsqueeze(0) // 255 for x in mask_serie]
        mask_serie = torch.cat(mask_serie)
        return arr_serie, mask_serie

        # if self.is_train:
        #     r0, r1, c0, c1 = RandomCrop(128)(arr_serie)
        #     return arr_serie[..., r0:r1, c0:c1], mask_serie[..., r0:r1, c0:c1]
        # else:
        #     return arr_serie, mask_serie

    def __len__(self):
        return len(self.series_id)
        
class IRCADSingle(data.Dataset):
    def __init__(self, root_path=None, is_train=True):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path 3D-IRCAD dataset!"
        self.root_path = Path(root_path)

        
        self.imgs_id = map(str, self.root_path.rglob('PATIENT_DICOM/image_[0-9]*'))

        regex = re.compile(r'.*3Dircadb1.(?P<patient_id>[0-9]+)/PATIENT_DICOM/image_(?P<slice_id>[0-9]+)')
        def get_key(x):
            groups = regex.match(x)
            patient_id = int(groups.group('patient_id'))
            slice_id = int(groups.group('slice_id'))
            return (patient_id, slice_id)
        
        self.imgs_id = sorted(self.imgs_id, key=get_key)
        self.is_train = is_train
        
    def __getitem__(self, index):
        img_id = self.imgs_id[index]

        dicom_img = dicom.dcmread(img_id)
        arr_img = dicom_img.pixel_array
        arr_img_tf = tvtf.Compose([
            tvtf.ToTensor(),
            NormMaxMin(),
        ])
        arr_img = arr_img_tf(arr_img)

        dicom_img = dicom.dcmread(img_id.replace('PATIENT_DICOM',
                                'MASKS_DICOM/liver'))
        mask_img = dicom_img.pixel_array
        mask_img_tf = tvtf.Compose([
        ])
        mask_img = torch.Tensor(mask_img_tf(mask_img))

        max_value = int(torch.max(mask_img))
        mask_img = mask_img.long() // (max_value if max_value > 0 else 1)
        
        # if self.is_train:
        #     r0, r1, c0, c1 = RandomCrop(128)(arr_img)
        #     return arr_img[..., r0:r1, c0:c1], mask_img[..., r0:r1, c0:c1]
        # else:
        #     return arr_img, mask_img

        return arr_img, mask_img

    def __len__(self):
        return len(self.imgs_id)
        
def test():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    dataset = IRCADSingle(root_path=args.root)

    for img, mask in dataset:
        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()
        plt.close()

def another_test():
    import argparse
    import matplotlib.pyplot as plt
    import torchvision as tv

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    dataset = IRCADSeries(root_path=args.root)
    
    for img, mask in dataset:
        plt.imshow(tv.utils.make_grid(img, nrow=15).permute(1, 2, 0))
        plt.imshow(tv.utils.make_grid(mask.unsqueeze(1), nrow=15).squeeze().permute(1, 2, 0) * 255, alpha=0.5)
        plt.tight_layout()
        plt.show()
        plt.close()
    
    
if __name__ == "__main__":
    another_test()