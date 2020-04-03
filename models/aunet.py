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
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        cfg = {
            'conv': {
                'padding': 0,
            },
            'activation': {
                'inplace': True,
            },
        }

        self.W_g = Conv2dBlock(F_g, F_int, kernel_size=1,
                               norm='none', activation='none', cfg=cfg)

        self.W_x = Conv2dBlock(F_l, F_int, kernel_size=1,
                                norm='none', activation='none', cfg=cfg)

        self.relu = nn.ReLU(inplace=True)

        self.phi = Conv2dBlock(F_int, 1, kernel_size=1,
                                norm='none', activation='none', cfg=cfg)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g_1 = self.W_g(g)
        x_1 = self.W_x(x)
        phi = self.relu(g_1 + x_1)
        phi = self.phi(phi)
        sigmoid = self.sigmoid(phi)
        return x*sigmoid

class EncoderBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.cfg = {
            'conv': {
                'padding': 1,
            },
            'activation': {
                'inplace': True,
            },
        }

        self.down_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.down_conv(x)
        pool = self.pool(x)
        return x, pool

class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs,
                upsample_method='deconv', sizematch_method='interpolate'):
        super().__init__()

        self.cfg = {
            'conv': {
                'padding': 1,
            },
            'activation': {
                'inplace': True,
            },
        }

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
        
        self.up_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg)
        )

        self.att = AttentionBlock(outputs, outputs, outputs//2)
    
    def sizematch_interpolate(self, source, target):
        return F.interpolate(source, size=(target.size(2), target.size(3)),
                             mode='bilinear', align_corners=True)

    def sizematch_pad(self, source, target):
        diffX = target.size()[3] - source.size()[3]
        diffY = target.size()[2] - source.size()[2]
        return F.pad(source, (diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffX - diffY // 2))

    def forward(self, x, x_copy):
        g = self.upsample(x)
        g = self.sizematch(g, x_copy)
        x_copy = self.att(g=g, x=x_copy)
        print(x_copy.shape)
        g = torch.cat([x_copy, g], dim=1)
        g = self.up_conv(g)
        #print(g.shape)
        return g

class AUNetMiddle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        cfg = {
            'conv': {
                'padding': 1,
            },
            'activation': {
                'inplace': True,
            },
        }

        self.conv = Conv2dBlock(in_channels, out_channels, kernel_size=3, cfg=cfg)

    def forward(self, x):
        x = self.conv(x)
        return x

class AUNetEncoder(nn.Module):
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

class AUNetDecoder(nn.Module):
    def __init__(self, depth, first_channels):
        super().__init__()
        levels = [DecoderBlock(first_channels // 2**i,
                                     first_channels // 2**(i+1))
                  for i in range(depth)]
        self.depth = depth
        self.levels = nn.ModuleList(levels)

    def forward(self, x, concats):
        for level, x_copy in zip(self.levels, concats):
            x = level(x, x_copy)
        return x

class AUNet(nn.Module):
    def __init__(self, nclasses, in_channels, depth):
        super().__init__()
        self.encoder = AUNetEncoder(in_channels, depth, 64)
        self.middle_conv = AUNetMiddle(64*2**(depth-1), 64*2**depth)
        self.decoder = AUNetDecoder(depth, 64*2**depth)
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
    net = AUNet(2, 3, 4)
    tbar = tqdm(range(2))
    for iter_id in tbar:
        inps = torch.rand(4, 3, 100, 100).to(dev)
        lbls = torch.randint(low=0, high=2, size=(4, 100, 100)).to(dev)

        outs = net(inps)
