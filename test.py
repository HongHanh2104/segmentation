from metrics.metrics import IoU
from models.toymodel import ToyModel
from workers.trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import yaml

if __name__ == "__main__":
    device = torch.device('cpu')
    path = 'configs/config.json'
    config = yaml.load(open(path), Loader=yaml.Loader)
    net = ToyModel(64, 13)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = optim.Adam(net.parameters())
    metric = IoU(nclasses=13, ignore_index=-1)
    trainer = Trainer(device, config, net, criterion, optimizer, metric)

    train_dataset = [(torch.randn(size=(3, 100, 100)),
                    torch.randn(size=(100, 100)),
                    torch.randint(low=-1, high=13, size=(100, 100)).long())
                    for _ in range(99)]
    train_dataloader = data.DataLoader(train_dataset, batch_size=4)

    test_dataset = [(torch.randn(size=(3, 100, 100)),
                    torch.randn(size=(100, 100)),
                    torch.randint(low=-1, high=13, size=(100, 100)).long())
                    for _ in range(99)]
    test_dataloader = data.DataLoader(test_dataset, batch_size=4)

    trainer.train(train_dataloader, test_dataloader)