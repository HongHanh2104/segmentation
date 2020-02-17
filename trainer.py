import torch
from torch import nn, optim
from torch.utils import data
from torchnet import meter
from tqdm import tqdm
import numpy as np
import os
import json

class Trainer():
    def __init__(self, device, 
                    config, 
                    net, 
                    criterion, 
                    optimier,
                    metric):
        super(Trainer, self).__init__()
        self.device = device
        self.net = net
        self.criterion = criterion
        self.optimizer = optimier
        self.metric = metric

        # Get arguments
        # self.batch_size = config["train"]["args"]["batch_size"]
        self.nepochs = config["train"]["args"]["epochs"]
        # self.num_workers = config["train"]["args"]["num_workers"]
        # self.num_iters = config["train"]["args"]["num_iters"]
        # self.save_period = config["train"]["args"]["save_period"]
        # self.early_stop = config["train"]["args"]["early_stop"]

        self.log_step = config["log"]["log_per_iter"]
        self.log_path = config["log"]["path"]
        self.best_loss = np.inf
        self.val_loss = []

    def save_checkpoint(self, epoch, val_loss):
        #best_loss = np.inf
        data = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        if val_loss < self.best_loss:
            print("Loss is improved from %.5f to %.5f. Saving weights ......" % (self.best_loss, val_loss))
            torch.save(data, os.path.join(self.log_path, "{}_best_loss.pth".format(epoch)))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print("Loss is not improved from %.6f." % (self.best_loss))

        # Save the model
        print("Saving curent model ....")
        torch.save(data, os.path.join(self.log_path, "{}.pth".format(epoch)))


    def train_epoch(self, epoch, dataloader): 
        # 0: Record loss during training process 
        running_loss = meter.AverageValueMeter()
        loss_metric = self.metric
        train_loss = []
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
            # 8: Update metric
            outs = outs.detach()
            label_imgs = label_imgs.detach()
            metric_value = loss_metric.calculate(outs, label_imgs)
            print(metric_value)
            # Update train loss
            train_loss.append(loss.item()) 

    
    def val_epoch(self, epoch, dataloader):
        running_loss = meter.AverageValueMeter()
        val_metric = self.metric
        self.net.eval()
        print("Validating ........")
        progress_bar = tqdm(dataloader)
        for i, (color_imgs, *label_imgs) in enumerate(progress_bar):
            # 1: Load inputs and labels
            color_imgs = color_imgs.to(self.device)
            depth_imgs = label_imgs[0].to(self.device)
            label_imgs = label_imgs[1].to(self.device)

            # 2: Get network outputs
            outs = self.net(color_imgs)

            # 3: Calculate the loss 
            loss = self.criterion(outs, label_imgs)

            # 4: Update loss
            running_loss.add(loss.item())
        
            # 5: Update metric
            #outs = outs.detach().cpu()
            #label_imgs = label_imgs.detach().cpu()
            metric_value = val_metric.IoU(outs, label_imgs, 13, -1)
        # 5: Get average loss 
        avg_loss = running_loss.value()[0]
        print("Average Loss: ", avg_loss)

        self.val_loss.append(avg_loss)
    
    def train(self, train_dataloader, val_dataloader):
        val_loss = 0
        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)
            
            # 2: Testing phase
            if (epoch + 1) % 5 == 0:
                self.val_epoch(epoch, dataloader=val_dataloader)
                print('-------------------------------')
             
            # 3: Learning rate scheduling

            # 4: Saving checkpoints
            if (epoch + 1) % 10 == 0:
                # Get latest val loss here 
                val_loss = self.val_loss[-1]

                self.save_checkpoint(epoch, val_loss)

            # 5: Visualizing some examples

if __name__ == "__main__":
    from metrics import IoU
    from toymodel import ToyModel

    device = torch.device('cpu')
    '''
    config = {
        "train": {
            "args": {
                "epochs": 5
            }
        }
    }
    '''
    path = 'config.json'
    config = json.load(open(path))
    net = ToyModel(64, 13)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = optim.Adam(net.parameters())
    metric = IoU(13, -1)
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
    