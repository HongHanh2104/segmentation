import torch 
import numpy as np

class Metrics():
    def __init__(self):
        super(Metrics, self).__init__()
        #self.output = output
        #self.target = target

    def IoU(self, output, target, nclasses):
        #print(output)
        ious = []
        _, prediction = torch.max(output, dim = 1)
        for c in range(nclasses):
            pred_c = prediction == c
            target_c = target == c
            intersection = torch.sum(pred_c & target_c)
            union = torch.sum(pred_c | target_c)
            '''
            if (intersection == 0 and union == 0):
                continue
            '''
            #print(intersection.float(), union.float(), intersection.float() / union.float())
            ious.append(intersection.float() + 1e-6 / union.float() + 1e-6)
            print(ious)
            
        miou = np.mean(ious)
        return miou
    


