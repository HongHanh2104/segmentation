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

def evaluate(config):
    dev_id = 'cuda:{}'.format(config['gpus']) \
            if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]
    output_dir = config['output']
    
    assert os.path.exists(pretrained_path)
    pretrained = torch.load(pretrained_path, map_location=dev_id)
    for item in ["model"]:
        config[item] = pretrained["config"][item]

    # Get model information
    num_class = config["model"]["num_class"]
    input_channel = config['model']['input_channel']
    method = config["model"]["method"]

    # 1: Load datasets
    # dataset = SUNRGBDDataset(root_path,
    #                                 img_folder,
    #                                 depth_folder,
    #                                 label_folder)

    dataset = IRCADSingle(root_path='data/3Dircadb1/test')
    dataloader = DataLoader(dataset, batch_size=1)

    # 2: Define network
    net = UNet(num_class, input_channel, method).to(device)
    net.load_state_dict(pretrained['model_state_dict'])
    
    # 5: Define metrics
    metric = IoU(nclasses=num_class)

    for idx, (inp, lbl) in enumerate(tqdm(dataloader)):
        # Get network output
        net.eval()
        start = time.time()
        out = net(inp.to(device)).detach()
        # print("Prediction time: %f" % (time.time()-start))
        
        # Post-process output for true prediction
        out = out.cpu()
        assert len(out.shape) in [3, 4]
        if len(out.shape) == 4: # BCHW
            conf, pred = torch.max(out, dim=1)
        elif len(out.shape) == 3: # BHW
            conf = torch.sigmoid(out)
            pred = (conf >= 0.5).long()
        pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8))

        # Plot prediction
        plt.figure(figsize=(30, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(inp.squeeze())
        plt.subplot(2, 2, 2)
        plt.imshow(lbl.squeeze(0))
        plt.subplot(2, 2, 3)
        plt.imshow(conf.squeeze(0))
        plt.subplot(2, 2, 4)
        plt.imshow(pred)
        plt.tight_layout()

        # Show plot (or save, depending on output_path specification)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'output_{idx:03d}'))
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--weight')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    config = dict()
    config['gpus'] = args.gpus
    config['pretrained'] = args.weight
    config['output'] = args.output

    evaluate(config)