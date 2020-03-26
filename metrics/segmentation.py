import torch
import torch.nn.functional as F
import numpy as np

from metrics.metrics import Metrics
from utils.segmentation import multi_class_prediction, binary_prediction


class PixelAccuracy(Metrics):
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.pred_fn = multi_class_prediction
        if nclasses == 1:
            self.nclasses += 1
            self.pred_fn = binary_prediction
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        prediction = self.pred_fn(output)

        image_size = target.size(1) * target.size(2)

        ignore_mask = torch.zeros(target.size()).bool().to(target.device)
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index).bool()
        ignore_size = ignore_mask.sum((1, 2))

        correct = ((prediction == target) | ignore_mask).sum((1, 2))
        acc = (correct - ignore_size + 1e-6) / \
            (image_size - ignore_size + 1e-6)
        return acc.cpu()

    def update(self, value):
        self.total_correct += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        return (self.total_correct / self.sample_size).item()

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0

    def summary(self):
        print(f'Pixel Accuracy: {self.value():.6f}')


class MeanIoU(Metrics):
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.pred_fn = multi_class_prediction
        if nclasses == 1:
            self.nclasses += 1
            self.pred_fn = binary_prediction
        self.ignore_index = ignore_index
        self.reset()

    def calculate(self, output, target):
        batch_size = output.size(0)
        ious = torch.zeros(self.nclasses, batch_size)

        prediction = self.pred_fn(output)

        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = self.ignore_index

        prediction = F.one_hot(prediction, self.nclasses).bool()
        target = F.one_hot(target, self.nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        ious = (intersection.float() + 1e-6) / (union.float() + 1e-6)

        return ious.cpu()

    def update(self, value):
        self.mean_class += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        return (self.mean_class / self.sample_size).mean()

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = self.mean_class / self.sample_size

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
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
