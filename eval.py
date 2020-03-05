import argparse 
import yaml
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torchnet import meter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets.sunrgbd import SUNRGBDDataset
from datasets.ircad import IRCADSingle
from models.unet import UNet
from workers.trainer import Trainer
from metrics.metrics import IoU
from utils.random_seed import set_seed

import time
import os

def train(config):
    assert config is not None, "Do not have config file!"
    
    dev_id = 'cuda:{}'.format(config['gpus']) \
            if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]
    output_dir = config['output']
    
    pretrained = None
    if (pretrained_path != None):
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model", "train"]:
            config[item] = pretrained["config"][item]

    # Get model information
    num_class = config["model"]["num_class"]
    input_channel = config['model']['input_channel']
    method = config["model"]["method"]

    # 1: Load datasets
    set_seed()
    
    # dataset = SUNRGBDDataset(root_path,
    #                                 img_folder,
    #                                 depth_folder,
    #                                 label_folder)
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    dataset = IRCADSingle(root_path='data/3Dircadb1/test')

    # 2: Define network
    set_seed()
    net = UNet(num_class, input_channel, method).to(device)
    # print(net)
    
    # 5: Define metrics
    metric = IoU(nclasses=num_class)

    # Train from pretrained if it is not None
    if (pretrained is not None):
        net.load_state_dict(pretrained['model_state_dict'])

    for idx, (inp, lbl) in enumerate(dataset):
        net.eval()
        start = time.time()
        out = net(inp.unsqueeze(0).to(device)).detach()
        print("Prediction time: %f" % (time.time()-start))
        
        out = out.cpu()
        assert len(out.shape) in [3, 4]
        if len(out.shape) == 4:
            _, pred = torch.max(out, dim=1)
        elif len(out.shape) == 3:
            pred = (out > 0).long()
        pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8))

        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(inp.squeeze())
        plt.subplot(1, 3, 2)
        plt.imshow(lbl)
        plt.subplot(1, 3, 3)
        plt.imshow(pred)
        plt.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'output_{idx:03d}'))
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--weight')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['pretrained'] = args.weight
    config['output'] = args.output

    train(config)