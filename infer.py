import os 
import argparse
from PIL import Image
import torch 
from torchvision import transforms
from models.toymodel import ToyModel
from models.unet import UNet
import time
import numpy as np

def read_image(img_path):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

@torch.no_grad()
def predict(net, device, image):
    net.eval()
    start = time.time()
    img = image.to(device)
    out = net(img).detach()
    _, pred = torch.max(out, dim=1) 
    print(pred.squeeze(0).shape)
    pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8))
    print("Prediction time: %f" % (time.time()-start))

    return pred


def infer(pretrained_path, input_path, output_path):
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # 1: Load pretrained net
    pretrained_weight = torch.load(pretrained_path, map_location=dev_id)
    net = ToyModel(64, 13)
    net.load_state_dict(pretrained_weight['model_state_dict'])
    
    # 2: Load image 
    input_img = read_image(input_path)

    # 3: Predict the image
    pred = predict(net=net,
                    device=device,
                    image=input_img)
    pred.save(os.path.join(output_path, 'prediction.png'))

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained')
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    infer(pretrained_path=args.pretrained,
            input_path=args.input,
            output_path=args.output)

if __name__ == "__main__":
    main()    

