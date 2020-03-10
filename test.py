import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.nn import functional as F

from datasets import IRCADSingle
from models import UNet
from losses import FocalLoss
from torch.nn import CrossEntropyLoss

# torch.random.manual_seed(3698)
# torch.backends.cudnn.deterministic = True

dataset = IRCADSingle('data/3Dircadb1/train', is_train=False)
dataloader = data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

model = UNet(2, 1, 'interpolate').cuda().eval()
model.load_state_dict(torch.load('backup/toyweights.pth'))
ce = CrossEntropyLoss()
fc = FocalLoss()

for img, lbl in dataloader:
    img = img.cuda()
    lbl = lbl.cuda()
    out = model(img)

    print(ce(out, lbl), fc(out, lbl))
    
    if out.dim()>2:
        out = out.view(out.size(0), out.size(1), -1)    # N,C,H,W => N,C,H*W
        out = out.transpose(1, 2)                       # N,C,H*W => N,H*W,C
        out = out.contiguous().view(-1, out.size(2))    # N,H*W,C => N*H*W,C
    lbl = lbl.view(-1,1) # N*H*W,1

    logpt = F.log_softmax(out, dim=1)
    logpt = logpt.gather(1, lbl)
    logpt = logpt.view(-1) # N*H*W,
    pt = Variable(logpt.data.exp())
    print(pt)

    break