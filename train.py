import argparse 
import yaml
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torchnet import meter

from datasets.sunrgbd import SUNRGBDDataset
from models.unet import UNet
from workers.trainer import Trainer
from metrics.metrics import IoU
from utils.random import set_seed

def train(config):
    assert config is not None, "Do not have config file!"
    
    dev_id = 'cuda:{}'.format(config['gpus']) \
            if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]
    
    pretrained = None
    if (pretrained_path != None):
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model", "train"]:
            config[item] = pretrained["config"][item]

    # Get model information
    num_class = config["model"]["num_class"]
    method = config["model"]["method"]

    # Get model args
    learning_rate = config["model"]["args"]["learning_rate"]
    momentum = config["model"]["args"]["momentum"]
    weight_decay = config["model"]["args"]["weight_decay"]
    
    # Get path
    root_path = config["train"]["path"]["root"]
    img_folder = config["train"]["path"]["img_folder"]
    depth_folder = config["train"]["path"]["depth_folder"]
    label_folder = config["train"]["path"]["label_folder"]

    # 1: Load datasets
    set_seed()
    train_dataset = SUNRGBDDataset(root_path,
                                    img_folder,
                                    depth_folder,
                                    label_folder)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                               [len(train_dataset) - len(train_dataset) // 5, len(train_dataset) // 5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=6, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=6, batch_size=1)

    # 2: Define network
    set_seed()
    net = UNet(num_class, method).to(device)
    print(net)
    
    # 3: Define loss
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    # 4: Define Optimizer & Scheduler 
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=learning_rate, 
                                momentum=momentum, 
                                weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=5, verbose=True)
    # 5: Define metrics
    metric = IoU(nclasses=num_class, ignore_index=-1)

    # Train from pretrained if it is not None
    if (pretrained is not None):
        net.load_state_dict(pretrained['model_state_dict'])
        optimizer.load_state_dict(pretrained['optimizer_state_dict'])

    # 6: Create trainer
    trainer = Trainer(device = device,
                        config = config,
                        net = net,
                        criterion = criterion,
                        optimier = optimizer,
                        scheduler = scheduler,
                        metric = metric)
    # 7: Start to train
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus

    train(config)