import torch
from torch import nn, optim
from torch.utils import data
from torchnet import meter
from tqdm import tqdm

from toymodel import ToyModel

class Trainer():
    def __init__(self, device, 
                    config, 
                    net, 
                    criterion, 
                    optimier):
        super(Trainer, self).__init__()
        self.device = device
        self.net = net
        self.criterion = criterion
        self.optimizer = optimier

        # Get arguments
        # self.batch_size = config["train"]["args"]["batch_size"]
        self.nepochs = config["train"]["args"]["epochs"]
        # self.num_workers = config["train"]["args"]["num_workers"]
        # self.num_iters = config["train"]["args"]["num_iters"]
        # self.save_period = config["train"]["args"]["save_period"]
        # self.early_stop = config["train"]["args"]["early_stop"]

    def train_epoch(self, epoch, dataloader): 
        # 0: Record loss during training process 
        running_loss = meter.AverageValueMeter()

        self.net.train()
        print("Training ........")
        progress_bar = tqdm(dataloader)
        for i, (color_imgs, *label_imgs) in enumerate(progress_bar):
            # 1: Load img_inputs and labels 
            color_imgs = color_imgs.to(self.device)
            depth_imgs = label_imgs[0].to(self.device)
            label_imgs = label_imgs[1].to(self.device)
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            # 3: Get network outputs 
            outs = self.net(color_imgs)
            # 4: Calculate the loss
            loss = self.criterion(outs, label_imgs)
            # 5: Calculate gradients
            loss.backward()
            # 6: Performing backpropagation
            self.optimizer.step()
            # 7: Update loss 
            running_loss.add(loss.item())
    
    def val_epoch(self, epoch):
        return 0
    
    def train(self, train_dataloader, test_dataloader):
        for epoch in range(self.nepochs):
            print('Epoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)
            
            # 2: Testing phase
            if epoch % 1 == 0:
                self.val_epoch(epoch)
                print('-------------------------------')
            # 3: Learning rate scheduling

            # 4: Saving checkpoints

            # 5: Visualizing some examples

if __name__ == "__main__":
    device = torch.device('cpu')
    config = {
        "train": {
            "args": {
                "epochs": 5
            }
        }
    }
    net = ToyModel(64, 20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(device, config, net, criterion, optimizer)

    train_dataset = [(torch.randn(size=(3, 100, 100)),
                    torch.randn(size=(100, 100)),
                    torch.randint(low=0, high=20, size=(100, 100)).long())
                    for _ in range(99)]
    train_dataloader = data.DataLoader(train_dataset, batch_size=4)

    test_dataset = [(torch.randn(size=(3, 100, 100)),
                    torch.randn(size=(100, 100)),
                    torch.randint(low=0, high=20, size=(100, 100)).long())
                    for _ in range(99)]
    test_dataloader = data.DataLoader(test_dataset, batch_size=4)

    trainer.train(train_dataloader, test_dataloader)
    