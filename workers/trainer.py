import torch
from torch import nn, optim
from torch.utils import data
from torchnet import meter
from tqdm import tqdm
import numpy as np
import os
import datetime

from loggers.tsboard import TensorboardHelper
from utils.debug import plot_grad_flow


class Trainer():
    def __init__(self, device,
                 config,
                 model,
                 criterion,
                 optimier,
                 scheduler,
                 metric):
        super(Trainer, self).__init__()
        self.config = config
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimier
        self.scheduler = scheduler
        self.metric = metric
        # Get arguments
        self.nepochs = self.config['trainer']['nepochs']
        self.val_step = self.config['trainer']['log']['val_step']
        self.best_loss = np.inf
        self.best_metric = 0.0
        self.val_loss = []
        self.val_metric = []
        self.save_dir = os.path.join(
            'runs', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.tsboard = TensorboardHelper(path=self.save_dir)

    def save_checkpoint(self, epoch, val_loss, val_metric):

        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }

        if val_loss < self.best_loss:
            print("Loss is improved from %.5f to %.5f. Saving weights ......" %
                  (self.best_loss, val_loss))
            torch.save(data, os.path.join(self.save_dir, "best_loss.pth"))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print("Loss is not improved from %.6f." % (self.best_loss))

        if val_metric > self.best_metric:
            print("Metric improved from %.6f to %.6f. Saving weights ..." %
                  (self.best_metric, val_metric))
            torch.save(data, os.path.join(self.save_dir, "best_metric.pth"))
            self.best_metric = val_metric
        else:
            print("Metric did not improve from %.6f ." % (self.best_metric))
        # Save the model
        print("Saving curent model ....")
        torch.save(data, os.path.join(
            self.save_dir, "current.pth".format(epoch)))
    def train_epoch(self, epoch, dataloader):
        # 0: Record loss during training process
        running_loss = meter.AverageValueMeter()
        self.metric.reset()
        self.model.train()
        print("Training ........")
        progress_bar = tqdm(dataloader)
        for i, (inp, lbl) in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            inp = inp.to(self.device)
            lbl = lbl.to(self.device)
            # label_imgs = label_imgs[1].to(self.device)
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            # 3: Get network outputs
            outs = self.model(inp)
            # 4: Calculate the loss
            loss = self.criterion(outs, lbl)
            # 5: Calculate gradients
            loss.backward()
            # plot_grad_flow(self.model.named_parameters(), f'epoch{epoch:02d}_iter{i:04d}')
            # 6: Performing backpropagation
            self.optimizer.step()
            # 7: Update loss
            running_loss.add(loss.item())
            self.tsboard.update_loss(
                'train', loss.item(), epoch * len(dataloader) + i)
            # 8: Update metric
            outs = outs.detach()
            lbl = lbl.detach()
            value = self.metric.calculate(outs, lbl)
            self.metric.update(value)
        print(self.metric.summary())

    @torch.no_grad()
    def val_epoch(self, epoch, dataloader):
        running_loss = meter.AverageValueMeter()
        self.metric.reset()
        self.model.eval()
        print("Validating ........")
        progress_bar = tqdm(dataloader)
        for i, (inp, lbl) in enumerate(progress_bar):
            # 1: Load inputs and labels
            inp = inp.to(self.device)
            lbl = lbl.to(self.device)

            # 2: Get network outputs
            outs = self.model(inp)

            # 3: Calculate the loss
            loss = self.criterion(outs, lbl)

            # 4: Update loss
            running_loss.add(loss.item())

            # 5: Update metric
            outs = outs.detach()
            lbl = lbl.detach()
            value = self.metric.calculate(outs, lbl)
            self.metric.update(value)

        # Get average loss
        avg_loss = running_loss.value()[0]
        print("Average Loss: ", avg_loss)
        self.val_loss.append(avg_loss)
        self.val_metric.append(self.metric.value())
        self.tsboard.update_loss('val', avg_loss, epoch)

        print(self.metric.summary())
        self.tsboard.update_metric('val', 'iou', self.metric.value(), epoch)

    def train(self, train_dataloader, val_dataloader):
        val_loss = 0
        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            # 2: Testing phase
            if (epoch + 1) % self.val_step == 0:
                self.val_epoch(epoch, dataloader=val_dataloader)
                print('-------------------------------')

            # 3: Learning rate scheduling
            self.scheduler.step(self.val_loss[-1])

            # 4: Saving checkpoints
            if (epoch + 1) % self.val_step == 0:
                # Get latest val loss here
                val_loss = self.val_loss[-1]
                val_metric = self.val_metric[-1]
                self.save_checkpoint(epoch, val_loss, val_metric)

            # 5: Visualizing some examples
