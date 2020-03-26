import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_channels, nfeatures, nclasses):
        super(ToyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, nfeatures, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cls = nn.Sequential(
            nn.Conv2d(nfeatures, nclasses, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.cls(x)
        return x


if __name__ == "__main__":
    dev = torch.device('cpu')
    net = ToyModel(128, 2).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for iter_id in range(100):
        inps = torch.rand(4, 3, 100, 100).to(dev)
        lbls = torch.randint(low=0, high=2, size=(4, 100, 100)).to(dev)

        outs = net(inps)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        print(outs.shape)
        print(iter_id, loss.item())
