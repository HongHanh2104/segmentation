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

class Conv2dLayer(nn.Module):
    def __init__(self, inputs, outputs):
        super(Conv2dLayer, self).__init__()

        self.cfg = {
            'conv': {'padding': 1},
        }

        self.down_conv = nn.Sequential(
            Conv2dBlock(inputs, outputs, kernel_size=3, cfg=self.cfg),
            Conv2dBlock(outputs, outputs, kernel_size=3, cfg=self.cfg),
        )

    def forward(self, x):
        # x shape [B, C, H, W]
        #print('Input:', x.shape)
        x = self.down_conv(x)  # down shape [B, outputs_channel, H, W]
        #print('Down:', x.shape)
        return x

class NestedUNet(nn.Module):
    def __init__(self, inp_channels, nclasses, first_channels):
        super(NestedUNet, self).__init__()

        self.conv0_0 = Conv2dLayer(inp_channels, first_channels) # (inp_channel, 64)
        self.conv1_0 = Conv2dLayer(first_channels, first_channels*2) # (64, 128)
        self.conv2_0 = Conv2dLayer(first_channels*2, first_channels*4)
        self.conv3_0 = Conv2dLayer(first_channels*4, first_channels*8)
        self.conv4_0 = Conv2dLayer(first_channels*8, first_channels*16)

        self.conv0_1 = Conv2dLayer(first_channels + first_channels*2, first_channels)
        self.conv1_1 = Conv2dLayer(first_channels*2 + first_channels*4, first_channels*2)
        self.conv2_1 = Conv2dLayer(first_channels*4 + first_channels*8, first_channels*4)
        self.conv3_1 = Conv2dLayer(first_channels*8 + first_channels*16, first_channels*8)

        self.conv0_2 = Conv2dLayer(first_channels*2 + first_channels*2, first_channels)
        self.conv1_2 = Conv2dLayer(first_channels*2*2 + first_channels*4, first_channels*2)
        self.conv2_2 = Conv2dLayer(first_channels*2*2*2 + first_channels*8, first_channels*4)

        self.conv0_3 = Conv2dLayer(first_channels*3 + first_channels*2, first_channels)
        self.conv1_3 = Conv2dLayer(first_channels*2*3 + first_channels*4, first_channels*2)

        self.conv0_4 = Conv2dLayer(first_channels*4 + first_channels*2, first_channels)

        self.final = nn.Conv2d(first_channels, nclasses, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def interpolate(self, x, x_copy):
        x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x0_0 = self.conv0_0(x) # [B, inp_channel, H, W]
        x1_0 = self.conv1_0(self.pool(x0_0)) # [B, inp_channel*2, H/2, W/2]
        x2_0 = self.conv2_0(self.pool(x1_0)) # [B, inp_channel*4, H/4, W/4]
        x3_0 = self.conv3_0(self.pool(x2_0)) # [B, inp_channel*8, H/8, W/8]
        x4_0 = self.conv4_0(self.pool(x3_0)) # [B, inp_channel*16, H/16, W/16]
        

        x0_1 = self.conv0_1(torch.cat([x0_0, 
                            self.interpolate(self.up(x1_0), x0_0)], 1)) # [B, inp_channel, H, W]
        x1_1 = self.conv1_1(torch.cat([x1_0, 
                            self.interpolate(self.up(x2_0), x1_0)], 1)) # [B, inp_channel*2, H/2, W/2]
        x2_1 = self.conv2_1(torch.cat([x2_0, 
                            self.interpolate(self.up(x3_0), x2_0)], 1)) # [B, inp_channel*4, H/4, W/4]
        x3_1 = self.conv3_1(torch.cat([x3_0, 
                            self.interpolate(self.up(x4_0), x3_0)], 1)) # [B, inp_channel*8, H/8, W/8]
        
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, 
                            self.interpolate(self.up(x1_1), x0_0)], 1)) # [B, inp_channel, H, W]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, 
                            self.interpolate(self.up(x2_1), x1_0)], 1)) # [B, inp_channel*2, H/2, W/2]
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, 
                            self.interpolate(self.up(x3_1), x2_0)], 1)) # [B, inp_channel*4, H/4, W/4]
        
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, 
                            self.interpolate(self.up(x1_2), x0_0)], 1)) # [B, inp_channel, H, W]
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, 
                            self.interpolate(self.up(x2_2), x1_0)], 1)) # [B, inp_channel*2, H/2, W/2]
        
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, 
                            self.interpolate(self.up(x1_3), x0_0)], 1)) # [B, inp_channel, H, W]
        
        print("x0_4: ", x0_4.shape)
        x = self.final(x0_4) 
        
        return x

if __name__ == "__main__":
    dev = torch.device('cpu')
    net = NestedUNet(3, 2, 64).to(dev)
    
    for iter_id in range(3):
        inps = torch.rand(4, 3, 100, 100).to(dev)
        outs = net(inps)
       

