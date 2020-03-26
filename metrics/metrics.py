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
        assert nclasses > 0
        self.binary = nclasses == 1
        self.nclasses = nclasses + self.binary
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        batch_size = output.size(0)
        ious = [[0 for _ in range(self.nclasses)] for _ in range(batch_size)]

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
        self.mean_class.extend(value)

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
