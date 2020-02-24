import torch 
import numpy as np
from ignite.metrics import ConfusionMatrix, IoU, mIoU


class Metrics():
    def __init__(self):
        pass

    def calculate(self, output, target):
        pass

    def update(self):
        pass

    def reset(self):
        pass 

    def value(self):
        pass

class IoU(Metrics):
    def __init__(self, nclasses, ignore_index):
        super().__init__()
        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.mean_class = {}

    def calculate(self, output, target):
        ious = {}
        _, prediction = torch.max(output, dim=1)
        target_mask = (target == self.ignore_index).bool()
        prediction[target_mask] = -1
        for c in range(self.nclasses):
            pred_c = prediction == c
            target_c = target == c
            intersection = torch.sum(pred_c & target_c)
            union = torch.sum(pred_c | target_c)
            iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
            ious[c] = iou.item()
        
        #miou = np.mean([iou for iou in self.mean_per_batch.values()])
        return ious

    def update(self, iou_per_batch):
        for c, value in iou_per_batch.items():
            self.mean_class[c] = value 
    
    def value(self):
        return np.mean([x for x in self.mean_class.values()])

    def reset(self):
        self.mean_class = {}

if __name__ == "__main__":
    nclasses = 3
    ignore_index = 0

    target = torch.Tensor([
        [
            [-1, 0],
            [2, 1]
        ],
        [
            [1, 2],
            [2, 1]
        ],
        [
            [1, -1],
            [-1, -1]
        ]
    ]).long()

    output = torch.Tensor([
        [
            [[1, 0], [0 , 1]],
            [[0, 1], [0 , 0]],
            [[0, 0], [1 , 0]]
        ],
        [
            [[1, 0], [0 , 1]],
            [[0, 1], [0 , 0]],
            [[0, 0], [1 , 0]]
        ],
        [
            [[1, 0], [0 , 1]],
            [[0, 1], [0 , 0]],
            [[0, 0], [1 , 0]]
        ]
    ]).float()

    metric = IoU(nclasses=nclasses, ignore_index=-1)
    print(metric.calculate(output, target))

    