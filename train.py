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
from metrics import Metric

import argparse

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
    metric = Metric(nclasses=num_class)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)

    args = parser.parse_args()
    config['gpus'] = args.gpus

    train(config)