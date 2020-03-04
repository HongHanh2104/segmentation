import torch 
import numpy as np

class Metrics():
    def __init__(self):
        pass

    def calculate(self, output, target):
        pass

    def update(self, value):
        pass

    def reset(self):
        pass 

    def value(self):
        pass

    def summary(self):
        pass

class IoU(Metrics):
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        ious = [0 for _ in range(self.nclasses)]
        
        if self.nclasses > 2:
            _, prediction = torch.max(output, dim=1)
        else:
            prediction = (output > 0).long()

        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = -1

        for c in range(self.nclasses):
            pred_c = prediction == c
            target_c = target == c
            intersection = torch.sum(pred_c & target_c)
            union = torch.sum(pred_c | target_c)
            iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
            ious[c] = iou.item() / output.size(0)

        return ious

    def update(self, iou_per_batch):
        self.mean_class.append(iou_per_batch)
    
    def value(self):
        return np.mean(self.mean_class)

    def reset(self):
        self.mean_class = []

    def summary(self):
        return np.mean(self.mean_class, axis=0)

if __name__ == "__main__":
    nclasses = 3
    ignore_index = -1

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

