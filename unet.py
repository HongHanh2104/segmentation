import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(EncoderBlock, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x = self.down_conv(x)
        pool = self.pool(x)
        return x, pool

class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs, method):
        super(DecoderBlock, self).__init__()
        self.method = method
        # Using deconv method
        self.up_transpose = nn.ConvTranspose2d(inputs, outputs, kernel_size = 2, stride = 2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )
    
    def forward(self, x, x_copy):
        x = self.up_transpose(x)
        if self.method == 'interpolate':
            x = F.interpolate(x, size = (x_copy.size(2), x_copy.size(3)),
                                mode = 'bilinear', align_corners = True)
        else:
            # for different sizes
            diffX = x_copy.size()[3] - x.size()[3]
            diffY = x_copy.size()[2] - x.size()[2]
            x = F.pad(x, (diffX // 2, diffX - diffX //2,
                        diffY // 2, diffX - diffY //2))

        # Concatenate
        x = torch.cat([x_copy, x], dim = 1)
        x = self.up_conv(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(MiddleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool

class UNet(nn.Module):
    def __init__(self, num_classes, method):
        super(UNet, self).__init__()
        
        self.down_conv1 = EncoderBlock(3, 64)
        self.down_conv2 = EncoderBlock(64, 128)
        self.down_conv3 = EncoderBlock(128, 256)
        self.down_conv4 = EncoderBlock(256, 512)

        self.middle_conv = MiddleBlock(512, 1024)

        self.up_conv1 = DecoderBlock(1024, 512, method)
        self.up_conv2 = DecoderBlock(512, 256, method)
        self.up_conv3 = DecoderBlock(256, 128, method)
        self.up_conv4 = DecoderBlock(128, 64, method)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size = 1)


    def forward(self, x):
        d_1, x = self.down_conv1(x)
        d_2, x = self.down_conv2(x)
        d_3, x = self.down_conv3(x)
        d_4, x = self.down_conv4(x)
        
        mid = self.middle_conv(x)
        
        u_1 = self.up_conv1(mid, d_4)
        u_2 = self.up_conv2(u_1, d_3)
        u_3 = self.up_conv3(u_2, d_2)
        u_4 = self.up_conv4(u_3, d_1)
        x = self.final_conv(u_4)
        return x
    
if __name__ == "__main__":
    dev = torch.device('cuda:0')
    net = UNet(2).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for iter_id in range(100):
        inps = torch.rand(4, 3, 224, 224).to(dev)
        lbls = torch.randint(low=0, high=2, size=(4, 224, 224)).to(dev)

        outs = net(inps)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()
        
        print(iter_id, loss.item())