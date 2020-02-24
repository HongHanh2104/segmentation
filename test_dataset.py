import matplotlib.pyplot as plt

from datasets.sunrgbd import SUNRGBDDataset

if __name__ == "__main__":
    dataset = SUNRGBDDataset(root_path = 'data/SUN-RGBD',
                    color_img_folder = 'SUNRGBD-train_images',
                    depth_img_folder = 'sunrgbd_train_depth',
                    label_img_folder = 'train13labels')
    
    for i, (color_img, depth_img, label_img) in enumerate(dataset):
        if i != 949: continue
        plt.subplot(3, 1, 1)
        plt.imshow(color_img.permute(1, 2, 0))
        plt.subplot(3, 1, 2)
        plt.imshow(label_img)
        plt.subplot(3, 1, 3)
        plt.imshow(depth_img.squeeze(0))
        plt.show()