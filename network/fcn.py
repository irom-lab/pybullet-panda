from torch import nn
import torch
import logging


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class FCN(nn.Module):
    def __init__(self, 
                 inner_channels=64, 
                 in_channels=1, 
                 out_channels=1, 
                 img_size=96,
                 bias=False,
                 verbose=True):
        super(FCN, self).__init__()

        # Downsample
        self.down_layer_1 = nn.Sequential(  # Nx1x96x96
            nn.Conv2d(
                in_channels=in_channels,  # depth only
                out_channels=inner_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
        )  # Nx(inner_channels//4)x48x48

        self.down_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels // 4,
                      out_channels=inner_channels // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(inner_channels // 2),
            nn.ReLU(),
        )  # Nx(inner_channels//2)x24x24

        self.down_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels // 2,
                      out_channels=inner_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
        )  # Nx(inner_channels)x12x12

        # Upsample
        self.up_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels,
                      out_channels=inner_channels // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(inner_channels // 2),
            nn.ReLU(),
            Interpolate(size=[img_size // 4, img_size // 4],
                        mode='bilinear'),  # Nx(inner_channels//2)x24x24
        )

        self.up_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels // 2,
                    #   + inner_channels // 2,
                      out_channels=inner_channels // 4,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
            Interpolate(size=[img_size // 2, img_size // 2],
                        mode='bilinear'),  # Nx(inner_channels//4)x48x48
        )

        self.up_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels // 4,
                    #   + inner_channels // 4,
                      out_channels=inner_channels // 8,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(inner_channels // 8),
            nn.ReLU(),
            Interpolate(size=[img_size, img_size],
                        mode='bilinear'),  # Nx(inner_channels//8)x96x96
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=inner_channels // 8, 
                    #   + in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),  # 1x1 convolution
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        if verbose:
            logging.info('Down layers:')
            logging.info(self.down_layer_1)
            logging.info(self.down_layer_2)
            logging.info(self.down_layer_3)
            logging.info('Up layers:')
            logging.info(self.up_layer_1)
            logging.info(self.up_layer_2)
            logging.info(self.up_layer_3)
            logging.info('Output layer:')
            logging.info(self.output_layer)


    def forward(self, x, append=None):
        down1 = self.down_layer_1(x)
        down2 = self.down_layer_2(down1)
        mid = self.down_layer_3(down2)
        up1 = self.up_layer_1(mid)
        # up1 = torch.cat((up1, down2), dim=1)
        up2 = self.up_layer_2(up1)
        # up2 = torch.cat((up2, down1), dim=1)
        up3 = self.up_layer_3(up2)
        # up3 = torch.cat((up3, x), dim=1)
        out = self.output_layer(up3)
        # print(out[0,0,30:33,27:30], torch.max(out), torch.min(out))
        return out
