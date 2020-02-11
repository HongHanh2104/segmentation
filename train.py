import argparse 
import json 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from sunrgbd import SUNRGBDDataset
from vgg import VGG16Feature

def train(config):
    root_path = config['train']['root_path']
    color_img_folder = config['train']['color_img_folder']
    depth_img_folder = config['train']['depth_img_folder']
    label_img_folder = config['train']['label']


    device = torch.device('cuda:{}'.format(config['gpus']) if torch.cuda.is_available() else 'cpu')

    # Load datasets
    training_set = SUNRGBDDataset(root_path,
                                color_img_folder,
                                depth_img_folder,
                                label_img_folder)
    training_loader = DataLoader(training_set, batch_size=4) # variable????

    # testing_set = SUNRGBDDataset(root_path = root_path.replace("train", "test"),
    #                             color_img_folder = color_img_folder.replace("train", "test"),
    #                             depth_img_folder = depth_img_folder.replace("train", "test"),
    #                             label_img_folder = label_img_folder.replace("train", "test"))
    
    # testing_loader = DataLoader(testing_set)

    net = VGG16Feature().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)

    num_iters = 10
    for iter_idx in range(num_iters):
        for batch_idx, (color_img, label_img, depth_img) in enumerate(dataloader):
            outs = net(color_img)
