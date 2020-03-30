import torch
import torch.nn as nn 
import torch.nn.functional as F

class RecurrentBlock(nn.Module):
    norm_map = {
        'none': nn.Identity,
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d    
    }

    activation_map = {
        'none': nn.Identity,
        'relu': nn.ReLU
    }
    
    def __init__(self, in_channels, 
                out_channels, 
                kernel_size,
                cfg, 
                norm='batch',
                activation='relu',
                t=2):
        super().__init__()
        self.t = t

        conv_cfg = {} if cfg.get('conv', None) is None else cfg['conv']
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True, **conv_cfg)

        assert norm in RecurrentBlock.norm_map.keys(), \
            "Chosen normalization method is not implemented."
        norm_cfg = {} if cfg.get('norm', None) is None else cfg['norm']
        self.norm = RecurrentBlock.norm_map[norm](out_channels, **norm_cfg)

        assert activation in RecurrentBlock.activation_map.keys(), \
            "Chosen normalization method is not implemented."
        activation_cfg = {} if cfg.get('activation', None) is None else cfg['activation']
        self.activation = RecurrentBlock.activation_map[activation](**activation_cfg)

    def forward(self, x):
        #print("x = ", x.shape)
        for i in range(self.t):
            if i == 0:
                out = self.norm(self.activation(self.conv(x)))
            out = self.norm(self.activation(self.conv(x + out)))
        #print("out = ", out.shape)
        return out
    
class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        
        self.cfg = {
            'conv': {
                'padding': 1,
            }
        }

        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                stride=1, padding=0)

        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, out_channels, kernel_size=3,
                            cfg=self.cfg, t=t),
            RecurrentBlock(out_channels, out_channels, kernel_size=3,
                            cfg=self.cfg, t=t)
        )
        #self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.RCNN(x1)
        #pool = self.pool(x1 + x2)
        return x1 + x2

class EncoderBlock(nn.Module):
    def __init__(self, inputs, outputs, t):
        super().__init__()
        self.conv = RRCNNBlock(inputs, outputs, t)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool

class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs, t, upsample_method='deconv',
                sizematch_method='interpolate'):
        super().__init__()

        assert upsample_method in ['deconv', 'interpolate']
        if upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                inputs, outputs, kernel_size=2, stride=2
            )
        elif upsample_method == 'interpolate':
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
        
        assert sizematch_method in ['interpolate', 'pad']
        if sizematch_method == 'interpolate':
            self.sizematch = self.sizematch_interpolate
        elif sizematch_method == 'pad':
            self.sizematch = self.sizematch_pad
        
        self.cfg = {
            'conv': {'padding': 1},
        }

        self.conv = RRCNNBlock(inputs, outputs, t)
    
    def sizematch_interpolate(self, source, target):
        return F.interpolate(source, size=(target.size(2), target.size(3)),
                             mode='bilinear', align_corners=True)

    def sizematch_pad(self, source, target):
        diffX = target.size()[3] - source.size()[3]
        diffY = target.size()[2] - source.size()[2]
        return F.pad(source, (diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffX - diffY // 2))

    def forward(self, x, x_copy):
        x = self.upsample(x)
        x = self.sizematch(x, x_copy)
        x = torch.cat([x_copy, x], dim=1)
        x = self.conv(x)
        return x


class R2UMiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()

        self.cfg = {
            'conv': {
                'padding': 1,
            }
        }

        self.conv = RRCNNBlock(in_channels, out_channels, t)
    
    def forward(self, x):
        x = self.conv(x)
        return x


class R2UEncoderBlock(nn.Module):
    def __init__(self, in_channels, depth, first_channels, t):
        super().__init__()
        levels = [EncoderBlock(in_channels, first_channels, t)]
        levels += [EncoderBlock(first_channels * 2**i,
                            first_channels * 2**(i+1), t)
                            for i in range(depth-1)]
        self.depth = depth
        self.levels = nn.ModuleList(levels)
        self.features = []
    
    def forward(self, x):
        self.features = []
        for i in range(self.depth):
            ft, x = self.levels[i](x)
            #print(ft.shape)
            self.features.append(ft)
        return x

    def get_features(self):
        return self.features[::-1]

class R2UDecoderBlock(nn.Module):
    def __init__(self, depth, first_channels, t):
        super().__init__()

        levels = [DecoderBlock(first_channels // 2**i,
                                first_channels // 2**(i+1), t)
                  for i in range(depth)]
        self.depth = depth
        self.levels = nn.ModuleList(levels)
        
    def forward(self, x, concats):
        for level, x_copy in zip(self.levels, concats):
            x = level(x, x_copy)
            #print(x.shape)
        return x

class R2UNet(nn.Module):
    def __init__(self, nclasses, in_channels, depth, t):
        super().__init__()
        self.encoder = R2UEncoderBlock(in_channels, depth, 64, t)
        self.middle_conv = R2UMiddleBlock(64*2**(depth-1), 64*2**depth, t)
        self.decoder = R2UDecoderBlock(depth, 64*2**depth, t)
        self.final_conv = nn.Conv2d(64, nclasses, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        features = self.encoder.get_features()
        mid = self.middle_conv(x)
        x = self.decoder(mid, features)
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    from tqdm import tqdm 
    dev = torch.device('cpu')
    net = R2UNet(2, 3, 4, 2).to(dev)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    tbar = tqdm(range(2))
    for i in tbar:
        inps = torch.rand(4, 3, 100, 100).to(dev)
        lbls = torch.randint(low=0, high=2, size=(4, 100, 100)).to(dev)

        outs = net(inps)

        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{i}: {loss.item()}')