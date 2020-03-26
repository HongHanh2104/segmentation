import torch
import numpy as np

from metrics.metrics import Metrics


class MeanIoU(Metrics):
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0
        self.binary = nclasses == 1
        self.nclasses = nclasses + self.binary
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        batch_size = output.size(0)
        ious = torch.zeros(batch_size, self.nclasses)

        if not self.binary:
            _, prediction = torch.max(output, dim=1)
        else:
            prediction = (output.squeeze(1) > 0).long()

        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = -1

        for i, (p, t) in enumerate(zip(prediction, target)):
            for c in range(self.nclasses):
                pred_c = p == c
                target_c = t == c
                intersection = torch.sum(pred_c & target_c)
                union = torch.sum(pred_c | target_c)
                iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
                ious[i][c] = iou.item()

        return ious

    def update(self, value):
        self.mean_class = self.mean_class * \
            self.sample_size + value.sum(0)
        self.sample_size += value.size(0)
        self.mean_class /= self.sample_size

    def value(self):
        return self.mean_class.mean()

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        for i, x in enumerate(self.mean_class):
            print(f'\tClass {i:3d}: {x:.6f}')


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
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ],
        [
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ],
        [
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ]
    ]).float()

    metric = IoU(nclasses=nclasses, ignore_index=-1)
    print(metric.calculate(output, target))
