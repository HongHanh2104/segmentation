import torch 
import numpy as np
from ignite.metrics import ConfusionMatrix, IoU, mIoU

'''
class Metrics():
    def __init__(self):
        pass

    def calculate(self, output, target):
        pass

    def update(self):
        pass

    def reset(self):
        pass 



class custom_IoU(Metrics):
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
        miou = np.mean([iou for iou in ious.values()])
        return miou
'''

class Metric():
    def __init__(self, nclasses=None):
        self.nclasses = nclasses
        self.value = None

    def calculate(self, matric_type, output, target):
        if matric_type == 'iou':
            self.value = self.mIoU(output, target).item()
        return self.value

    def IoU(self, output, target):
        assert self.nclasses != None, "nclasses is None"
        cm = ConfusionMatrix(num_classes=self.nclasses)
        iou_metric = IoU(cm)
        out = (output, target)
        cm.update(out)
        res = iou_metric.compute().numpy()
        return res
    
    def mIoU(self, output, target):
        assert self.nclasses != None, "nclasses is None"
        cm = ConfusionMatrix(num_classes=self.nclasses)
        miou_metric = mIoU(cm)
        out = (output, target)
        cm.update(out)
        res = miou_metric.compute().numpy()
        return res


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

    metric = Metric(nclasses=nclasses)
    print(type(metric.calculate('iou', output, target)))

    '''
    cm = ConfusionMatrix(num_classes=3)
    iou_metric = IoU(cm)

    result = (output, target)
    cm.update(result)

    res = iou_metric.compute().numpy()
    '''
    '''
    meter = custom_IoU(nclasses, ignore_index)

    for lbl, out in zip(target, output):
        lbl = lbl.unsqueeze(0)
        out = out.unsqueeze(0)
        ious = meter.calculate(out, lbl)
        print(ious)
    '''