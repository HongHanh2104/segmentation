import torch 
import numpy as np

class Metrics():
    def __init__(self):
        pass

    def calculate(self, output, target):
        pass

class IoU(Metrics):
    def __init__(self, nclasses, ignore_index):
        super().__init__()
        self.nclasses = nclasses
        self.ignore_index = ignore_index

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
        return ious
    
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

    meter = IoU(nclasses, ignore_index)

    for lbl, out in zip(target, output):
        lbl = lbl.unsqueeze(0)
        out = out.unsqueeze(0)
        ious = meter.calculate(out, lbl)
        print(ious)