import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchnet import meter

from workers.trainer import Trainer
from metrics.metrics import IoU
from utils.random_seed import set_seed
from losses import *
from datasets import *
from models import *

import argparse


def get_instance(config, **kwargs):
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)


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

    # 1: Load datasets
    set_seed()
    train_dataset = get_instance(config['dataset']['train'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   **config['dataset']['train']['loader'])

    val_dataset = get_instance(config['dataset']['val'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 **config['dataset']['val']['loader'])

    # 2: Define network
    set_seed()
    model = get_instance(config['model']).to(device)
    print(model)

    # Train from pretrained if it is not None
    if pretrained is not None:
        model.load_state_dict(pretrained['model_state_dict'])

    # 3: Define loss
    criterion = get_instance(config['loss']).to(device)

    # 4: Define Optimizer
    optimizer = get_instance(config['optimizer'],
                             params=model.parameters())
    if pretrained is not None:
        optimizer.load_state_dict(pretrained['optimizer_state_dict'])

    # 5: Define Scheduler
    scheduler = get_instance(config['scheduler'],
                             optimizer=optimizer)

    # 6: Define metrics
    metric = get_instance(config['metric'][0])

    # 6: Create trainer
    trainer = Trainer(device=device,
                      config=config,
                      model=model,
                      criterion=criterion,
                      optimier=optimizer,
                      scheduler=scheduler,
                      metric=metric)

    # 7: Start to train
    set_seed()
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus

    train(config)
