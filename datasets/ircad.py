import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pydicom as dicom
import glob
from utils.utils import rescale, NormMaxMin
import matplotlib.pyplot as plt

def split_string(ori_str, split_str):
    assert len(ori_str) > len(split_str), "Len of original string with split string are not suitable!"
    return ori_str.split(split_str)[1]

class IRCAD(data.Dataset):
    def __init__(self, root_path = None, type = 'image'):
        super().__init__()
        assert root_path is not None, "Missing root path, should be a path 3D-IRCAD dataset!"
        self.type = type
        self.root_path = root_path
        self.imgs_id = []
        for name in glob.glob(os.path.join(self.root_path, "3Dircadb1.[0-9]*/PATIENT_DICOM/image_[0-9]*")):
            self.imgs_id.append(name)

        '''
        folder_list = [x for x in sorted(os.listdir(self.root_path))]
        self.imgs_id = {}
        for i in range(len(folder_list)):
            self.image_path = os.path.join(self.root_path + "/" + folder_list[i], "PATIENT_DICOM")
            self.mask_path = os.path.join(self.root_path + "/" + folder_list[i], "MASKS_DICOM" + "/" + object)
            self.imgs_id[split_string(folder_list[i], "3Dircadb")] = [x for x in sorted(os.listdir(self.image_path))]
        '''
        
    def __getitem__(self, index):
        img_id = self.imgs_id[index]

        dicom_img = dicom.dcmread(img_id)
        arr_img = dicom_img.pixel_array

        arr_img_tf = transforms.Compose([
            transforms.ToTensor(),
            NormMaxMin()
        ])
        arr_img = arr_img_tf(arr_img)
        #plt.imshow(arr_img.squeeze(0))
        #plt.show()

        dicom_img = dicom.dcmread(img_id.replace('PATIENT_DICOM',
                                'MASKS_DICOM/liver'))
        mask_img = dicom_img.pixel_array
        mask_img_tf = transforms.Compose([
        ])
        mask_img = mask_img_tf(mask_img)
        
        return arr_img, mask_img
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--type", default="image")
    args = parser.parse_args()
    dataset = IRCAD(root_path=args.root, type=args.type)
    dataset.__getitem__(80)

if __name__ == "__main__":
    main()



