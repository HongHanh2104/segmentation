import os 
import argparse
from PIL import Image
import torch 
from torchvision import transforms
from models.toymodel import ToyModel
from models.unet import UNet
import time
import numpy as np
import matplotlib.pyplot as plt

from utils.image import load_image_as_tensor

@torch.no_grad()
def predict(net, inp):
    net.eval()
    start = time.time()
    out = net(inp).detach()
    print("Prediction time: %f" % (time.time()-start))
    return out

def post_process(out):
    out = out.cpu()
    _, pred = torch.max(out, dim=1)
    return Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8))

def infer(pretrained_path, input_path, output_path=None, gpus=None):
    dev_id = f'cuda:{gpus}' if torch.cuda.is_available() \
                               and gpus is not None \
             else 'cpu'
    device = torch.device(dev_id)

    # 1: Load pretrained net
    pretrained_weight = torch.load(pretrained_path, map_location=dev_id)
    net = ToyModel(64, 13).to(device)
    net.load_state_dict(pretrained_weight['model_state_dict'])
    
    # 2: Load image 
    input_img = load_image_as_tensor(input_path).unsqueeze(0).to(device)

    # 3: Predict the image
    out = predict(net=net,
                  inp=input_img)

    pred = post_process(out)

    if output_path is not None:
        pred.save(os.path.join(output_path, 'prediction.png'))
    else:
        plt.imshow(pred)
        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained')
    parser.add_argument('--input')
    parser.add_argument('--output', default=None)
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    infer(pretrained_path=args.pretrained,
          input_path=args.input,
          output_path=args.output,
          gpus=args.gpus)

if __name__ == "__main__":
    main()