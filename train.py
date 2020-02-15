import argparse 
import json 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from sunrgbd import SUNRGBDDataset
from unet import UNet
from tqdm import tqdm
from torchnet import meter 
from trainer import Trainer
from metrics import Metrics

import sys

def train(config):
    assert config is not None, "Do not have config file!"

    device = torch.device('cuda:{}'.format(config['gpus']) if config.get('gpus', False) and torch.cuda.is_available() else 'cpu')

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

    # 1: Load datasets
    train_dataset = SUNRGBDDataset(root_path,
                                    img_folder,
                                    depth_folder,
                                    label_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    val_dataset = SUNRGBDDataset(root_path,
                                    img_folder,
                                    depth_folder,
                                    label_folder)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    # 2: Define network
    net = UNet(num_class, method).to(device)
    print(net)
    # 3: Define loss
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    # 4: Define Optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=learning_rate, 
                                momentum=momentum, 
                                weight_decay=weight_decay)
    # 5: Define metrics
    metric = Metrics()

    # 6: Create trainer
    trainer = Trainer(device = device,
                        config = config,
                        net = net,
                        criterion = criterion,
                        optimier = optimizer,
                        metric = metric)
    # 7: Start to train
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

if __name__ == "__main__":
    config_path = 'config.json'
    config = json.load(open(config_path, 'r'))
    train(config)

# if __name__ == "__main__":
#     dev = torch.device('cuda:0')
#     net = UNet(13, 'interpolate').to(dev)
#     criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
#     optimizer = torch.optim.Adam(net.parameters())

#     num_iters = 100

#     dataset = SUNRGBDDataset(root_path = 'data',
#                     color_img_folder = 'SUNRGBD-train_images',
#                     depth_img_folder = 'sunrgbd_train_depth',
#                     label_img_folder = 'train13labels')
#     dataloader = data.DataLoader(dataset, batch_size=int(sys.argv[1]))

#     min_loss = 1000000
#     loss_log_step = 100
#     for iter_idx in range(num_iters):
#         print('Iter #{}: '.format(iter_idx))
#         running_loss = 0.0
#         total_loss = 0.0
#         for batch_idx, (color_img, depth_img, label_img) in enumerate(dataloader):
#             inps = color_img.to(dev)
#             lbls = label_img.to(dev)
#             optimizer.zero_grad()
#             outs = net(inps)
#             loss = criterion(outs, lbls)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() / len(dataloader)
#             running_loss += loss.item() / loss_log_step
#             if batch_idx % loss_log_step == 0:
#                 print(running_loss)
#                 running_loss = 0.0
#         if total_loss < min_loss:
#             torch.save(net.state_dict(), 'weights/save.pth')
#             min_loss = running_loss
