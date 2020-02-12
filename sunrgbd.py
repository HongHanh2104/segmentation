import torch
from torchvision import transforms as tvtf
from torch.utils import data
import os 
import numpy as np
from PIL import Image
from utils import execute_filename, rescale, NormMaxMin
import matplotlib.pyplot as plt

class SUNRGBDDataset(data.Dataset):
    def __init__(self,  root_path = None,
                        color_img_folder = None,
                        depth_img_folder = None,
                        label_img_folder = None):
        super(SUNRGBDDataset, None).__init__()
        
        assert root_path is not None, "Missing root path, should be a path to SUNRGBD dataset!"
        self.root_path = root_path
        
        assert color_img_folder is not None, "Missing color dataset!"
        self.color_img_path = os.path.join(self.root_path, color_img_folder)

        assert depth_img_folder is not None, "Missing depth dataset!"
        self.depth_img_path = os.path.join(self.root_path, depth_img_folder)

        assert label_img_folder is not None, "Missing label dataset!"
        self.label_img_path = os.path.join(self.root_path, label_img_folder)

        self.img_ids = [execute_filename(os.path.splitext(x)[0]) for x in sorted(os.listdir(self.color_img_path))]
                
    def __getitem__(self, index):
        img_id = self.img_ids[index]

        color_img = Image.open(os.path.join(self.color_img_path, 'img-' + img_id) + '.jpg').convert('RGB')
        color_img_tf = tvtf.Compose([
            #tvtf.Resize((224, 224)),
            tvtf.ToTensor()
        ])
        color_img = color_img_tf(color_img)

        label_img = Image.open(os.path.join(self.label_img_path, 'img13labels-' + img_id) + '.png')
        label_img_tf = tvtf.Compose([
        ])
        label_img = label_img_tf(label_img)
        label_img = torch.Tensor(np.array(label_img)).long()

        depth_img = Image.open(os.path.join(self.depth_img_path, str(int(img_id))) + '.png')
        depth_img_tf = tvtf.Compose([
            tvtf.ToTensor(),
            NormMaxMin()
        ])
        depth_img = depth_img_tf(depth_img)

        return color_img, depth_img, label_img

    def __len__(self):
        return len(self.img_ids)

if __name__ == "__main__":
    dataset = SUNRGBDDataset(root_path = 'data',
                    color_img_folder = 'SUNRGBD-train_images',
                    depth_img_folder = 'sunrgbd_train_depth',
                    label_img_folder = 'train13labels')
    
    for i, (color_img, depth_img, label_img) in enumerate(dataset):
        plt.subplot(3, 1, 1)
        plt.imshow(color_img.permute(1, 2, 0))
        plt.subplot(3, 1, 2)
        plt.imshow(label_img)
        plt.subplot(3, 1, 3)
        plt.imshow(depth_img.squeeze(0))
        plt.show()
        # break