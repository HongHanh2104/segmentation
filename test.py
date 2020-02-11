from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

path = '/media/honghanh/STUDY/DOCUMENT/MY_SWEET/MY_PROJECT/Thesis/Segmentation/Data/SUN RGBD/sunrgb_train_depth/sunrgbd_train_depth/1.png'

img = Image.open(path)

(min_pixel, max_pixel) = img.getextrema()

img_np = np.array(img, dtype = float)

img_np_ = (img_np - np.min(img_np))/(np.max(img_np) - np.min(img_np))
#plt.plot(img_np_)
plt.imshow(img_np_)
plt.show()