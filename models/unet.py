import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    norm_map = {
        'none': nn.Identity,
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
    }

    activation_map = {
        'none': nn.Identity,
        'relu': nn.ReLU,
    }

    def __init__(self, in_channels, out_channels, kernel_size, cfg,
                 norm='batch', activation='relu'):
        super().__init__()

        conv_cfg = {} if cfg.get('conv', None) is None else cfg['conv']
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, **conv_cfg)

        assert norm in Conv2dBlock.norm_map.keys(), \
            'Chosen normalization method is not implemented.'
        norm_cfg = {} if cfg.get('norm', None) is None else cfg['norm']
        self.norm = Conv2dBlock.norm_map[norm](out_channels, **norm_cfg)

        assert activation in Conv2dBlock.activation_map.keys(), \
            'Chosen activation method is not implemented.'
        activation_cfg = {} if cfg.get(
            'activation', None) is None else cfg['activation']
        self.activation = Conv2dBlock.activation_map[activation](
            **activation_cfg)

        self.skipable = in_channels == out_channels

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class EncoderBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(EncoderBlock, self).__init__()

        self.cfg = {
            'conv': {'padding': 1},
        }

        self.down_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        #print('Input:', x.shape)
        x = self.down_conv(x)
        pool = self.pool(x)
        #print('Down:', x.shape, pool.shape)
        return x, pool


class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs, method):
        super(DecoderBlock, self).__init__()
        self.method = method
        # Using deconv method
        self.up_transpose = nn.ConvTranspose2d(
            inputs, outputs, kernel_size=2, stride=2)

        self.cfg = {
            'conv': {'padding': 1},
        }

        self.up_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )

    def forward(self, x, x_copy):
        #print('Input:', x.shape, x_copy.shape)
        x = self.up_transpose(x)
        #print('Up:', x.shape)
        if self.method == 'interpolate':
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode='bilinear', align_corners=True)
        else:
            # for different sizes
            diffX = x_copy.size()[3] - x.size()[3]
            diffY = x_copy.size()[2] - x.size()[2]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffX - diffY // 2))
        #print('Scale:', x.shape)

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        #print('Concat', x.shape)
        x = self.up_conv(x)
        #print('UpConv:', x.shape)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(MiddleBlock, self).__init__()
        self.cfg = {
            'conv': {'padding': 1},
        }

        self.conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )

    def forward(self, x):
        #print('Input:', x.shape)
        x = self.conv(x)
        #print('Middle:', x.shape)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, depth, first_channels):
        super().__init__()

        levels = [EncoderBlock(in_channels, first_channels)]
        levels += [EncoderBlock(first_channels * 2**i,
                                first_channels * 2**(i+1))
                   for i in range(depth-1)]

        self.depth = depth
        self.levels = nn.ModuleList(levels)
        self.features = []

    def forward(self, x):
        self.features = []
        for i in range(self.depth):
            ft, x = self.levels[i](x)
            self.features.append(ft)
        return x

    def get_features(self):
        return self.features[::-1]


class UNetDecoder(nn.Module):
    def __init__(self, depth, first_channels):
        super().__init__()

        levels = [DecoderBlock(first_channels // 2**i,
                               first_channels // 2**(i+1),
                               '')
                  for i in range(depth)]

        self.depth = depth
        self.levels = nn.ModuleList(levels)

    def forward(self, x, concats):
        for level, x_copy in zip(self.levels, concats):
            x = level(x, x_copy)
        return x


class UNet(nn.Module):
    def __init__(self, nclasses, in_channels, depth):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, depth, 64)
        self.middle_conv = MiddleBlock(64 * 2**(depth - 1), 64 * 2**depth)
        self.decoder = UNetDecoder(depth, 64 * 2**depth)
        self.final_conv = nn.Conv2d(64, nclasses, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        features = self.encoder.get_features()

        mid = self.middle_conv(x)

        x = self.decoder(mid, features)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    dev = torch.device('cuda')
    net = UNet(2, 3, 4).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    for iter_id in range(100):
        inps = torch.rand(4, 3, 100, 100).to(dev)
        lbls = torch.randint(low=0, high=2, size=(4, 100, 100)).to(dev)

        outs = net(inps)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        print(iter_id, loss.item())
