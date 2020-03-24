import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.nn import functional as F

from datasets import IRCADSingle, IRCADSeries
from models.toymodel import ToyModel
from losses import FocalLoss
from torch.nn import CrossEntropyLoss

# torch.random.manual_seed(3698)
# torch.backends.cudnn.deterministic = True

dataset = IRCADSeries('data/3Dircadb1/train', is_train=False)
dataloader = data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

# model = UNet(2, 1, 'interpolate').cuda().eval()
# model.load_state_dict(torch.load('backup/UNet2D-Single-CE-Adam-5e6_2020-03-06_07_06_50/best_metric.pth')['model_state_dict'])
model = ToyModel(1, 256, 2).cuda().eval()
ce = CrossEntropyLoss()
fc = FocalLoss(gamma=5, alpha=15)

for img, lbl in dataloader:    
    img = img.cuda()
    lbl = lbl.cuda()
    print(img.shape, lbl.shape)
    input()

    for x, y in zip(img[0,:-1], img[0,1:]):
        x = x.unsqueeze(0)
        
        x = model.conv(x)