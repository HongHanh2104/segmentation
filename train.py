import argparse 
import json 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from sunrgbd import SUNRGBDDataset
from unet import UNet
from tqdm import tqdm


class Trainer():
    def __init__(self, device, 
                    config, 
                    net, 
                    criterion, 
                    optimier, 
                    train_dataloader,
                    test_dataloader):
        super(Trainer, self).__init__()
        self.device = device
        self.net = net
        self.criterion = criterion
        self.optimizer = optimier
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Get arguments
        self.batch_size = config["train"]["args"]["batch_size"]
        self.epochs = config["train"]["args"]["epochs"]
        self.num_workers = config["train"]["args"]["um_workers"]
        self.num_iters = config["train"]["args"]["num_iters"]
        self.save_period = config["train"]["args"]["save_period"]
        self.early_stop = config["train"]["args"]["early_stop"]

    def train_epoch(self, epoch):
        self.net.train()
        print("Training ........")
        progress_bar = tqdm(self.dataloader)
        for i, (color_imgs, depth_imgs, label_imgs) in enumerate(progress_bar):
            color_imgs = color_imgs.to(self.device)
            depth_imgs = depth_imgs.to(self.device)
            label_imgs = label_imgs.to(self.device)

            self.optimizer.zero_grad()
            outs = self.net(color_imgs)
            loss = self.criterion(outs, color_imgs)
            # Calculate gradients
            loss.backward()
            # Performing backpropagation
            self.optimizer.step()
            self.total_loss.update(loss.item())
    
    def val_epoch(self, epoch):
        return 0
    
    def train(self):
        for epoch in range(self.epochs):
            print('Epoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            self.train_epoch(epoch = epoch)
            
            if epoch % 1 == 0:
                self.val_epoch(epoch)
                print('-------------------------------')

def train(config):
    assert config is not None, "Do not have config file!"

    device = torch.device('cuda:{}'.format(config['gpus']) if torch.cuda.is_available() else 'cpu')

    # Get model information
    model_type = config["model"]["name"]
    num_class = config["model"]["num_class"]
    method = config["model"]["method"]

    # Get model args
    learning_rate = config["model"]["args"]["learning_rate"]
    momentum = config["model"]["args"]["momentum"]
    weight_decay = config["model"]["args"]["weight_decay"]

    # Get dataset information
    dataset_type = config["train"]["type"]
    
    # Get path
    root_path = config["train"]["path"]["root"]
    img_folder = config["train"]["path"]["img_folder"]
    depth_folder = config["train"]["path"]["depth_folder"]
    label_folder = config["train"]["path"]["label_folder"]
    save_path = config["train"]["path"]["save_path"]

    # Load datasets
    training_set = SUNRGBDDataset(root_path,
                                    img_folder,
                                    depth_folder,
                                    label_folder)
    training_loader = DataLoader(training_set, batch_size = 4)

    # testing_set = SUNRGBDDataset(root_path = root_path.replace("train", "test"),
    #                             color_img_folder = color_img_folder.replace("train", "test"),
    #                             depth_img_folder = depth_img_folder.replace("train", "test"),
    #                             label_img_folder = label_img_folder.replace("train", "test"))
    
    # testing_loader = DataLoader(testing_set)

    # Define network
    net = UNet(num_class, method).to(device)
    print(net)
    # Define loss
    criterion = nn.CrossEntropyLoss()
    # Define Optim
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)

    # Create trainer
    trainer = Trainer(device = device,
                        config = config,
                        net = net,
                        criterion = criterion,
                        optimier = optimizer,
                        dataloader = DataLoader)



    num_iters = 10
    for iter_idx in range(num_iters):
        for batch_idx, (color_img, label_img, depth_img) in enumerate(dataloader):
            outs = net(color_img)

if __name__ == "__main__":
    dev = torch.device('cuda:0')
    net = UNet(13, 'interpolate').to(dev)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = torch.optim.Adam(net.parameters())

    num_iters = 100

    dataset = SUNRGBDDataset(root_path = 'data',
                    color_img_folder = 'SUNRGBD-train_images',
                    depth_img_folder = 'sunrgbd_train_depth',
                    label_img_folder = 'train13labels')
    dataloader = data.DataLoader(dataset, batch_size=1)

    min_loss = 1000000
    for iter_idx in range(num_iters):
        print('Iter #{}: '.format(iter_idx))
        running_loss = 0.0
        for batch_idx, (color_img, depth_img, label_img) in enumerate(dataloader):
            inps = color_img.to(dev)
            lbls = label_img.to(dev)

            optimizer.zero_grad()
            outs = net(inps)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(dataloader)
        if running_loss < min_loss:
            torch.save(net.state_dict(), 'weights/save.pth')