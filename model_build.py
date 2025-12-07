import torch
import torch.nn as nn

CYCLEGAN_GEN_CONFIG = {
    "DownConvBlock": [
        (3, 64, 0.0, False),
        (64, 128, 0.0, True),
        (128, 256, 0.0, True),
        (256, 512, 0.5, True),
        (512, 512, 0.5, True),
        (512, 512, 0.5, True),
        (512, 512, 0.5, True),
        (512, 512, 0.5, False)
    ],
    "UpConvBlock": [
        (512, 512, 0.5),
        (1024, 512, 0.5),
        (1024, 512, 0.5),
        (1024, 512, 0.5),
        (1024, 256, 0.0),
        (512, 128, 0.0),
        (256, 64, 0.0)
    ],
    "UpSample": 2,
    "ZeroPad2d": (1, 0, 1, 0),
    "final_conv": (128, 3, 4, 1, 1)
}

CYCLEGAN_DSC_CONFIG = {
    "disc_conv_block": [
        (3, 64, False),
        (64, 128, True),
        (128, 256, True),
        (256, 512, True)
    ],
    "ZeroPad2d": (1, 0, 1, 0),
    "final_conv": (512, 1, 4, 1, 1)
}


class DownConvBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, norm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(input_size, output_size, 4, 2, 1)
        ])
        if norm: self.layers.append(nn.InstanceNorm2d(output_size))
        self.layers.append(nn.LeakyReLU(0.2))
        if dropout: self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        res = nn.Sequential(*self.layers)(x)
        return res
    

class UpConvBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(input_size, output_size, 4, 2, 1),
            nn.InstanceNorm2d(output_size),
            nn.ReLU()
        ])
        if dropout: self.layers.append(nn.Dropout(dropout))

    def forward(self, x, encoding_input):
        x = nn.Sequential(*self.layers)(x)
        res = torch.cat((x, encoding_input), dim=1)
        return res
    

class CycleGANGenerator(nn.Module):
    def __init__(self, config=CYCLEGAN_GEN_CONFIG):
        super(CycleGANGenerator, self).__init__()
        
        self.downconv_layers = nn.ModuleList([])
        for layer in config.get("DownConvBlock", []):
            self.downconv_layers.append(DownConvBlock(*layer))

        self.upconv_layers = nn.ModuleList([])
        for layer in config.get("UpConvBlock", []):
            self.upconv_layers.append(UpConvBlock(*layer))

        self.final_layers = nn.ModuleList([
            nn.Upsample(scale_factor=config["UpSample"]),
            nn.ZeroPad2d(padding=config["ZeroPad2d"]),
            nn.Conv2d(*config["final_conv"]),
            nn.Tanh()
        ])

    def forward(self, x):
        encs = [x]
        m = len(self.downconv_layers)
        n = len(self.upconv_layers)

        if self.downconv_layers:
            for i in range(m):
                encs.append(self.downconv_layers[i](encs[i]))

        res = encs[-1]
        if self.upconv_layers and len(encs) >= 2:
            for i in range(n):
                res = self.upconv_layers[i](res, encs[n-i])

        res = nn.Sequential(*self.final_layers)(res)
        return res
    

class CycleGANDiscriminator(nn.Module):
    def __init__(self, config=CYCLEGAN_DSC_CONFIG):
        super(CycleGANDiscriminator, self).__init__()

        def disc_conv_block(input_size, output_size, norm=True):
            mods = nn.ModuleList([
                nn.Conv2d(input_size, output_size, 4, 2, 1)
            ])
            if norm: mods.append(nn.InstanceNorm2d(output_size))
            mods.append(nn.LeakyReLU(0.2, inplace=True))
            return mods
        
        self.layers = nn.ModuleList([])
        for layer in config.get("disc_conv_block", []):
            mods = disc_conv_block(*layer)
            for mod in mods:
                self.layers.append(mod)

        self.layers.append(nn.ZeroPad2d(padding=config["ZeroPad2d"]))
        self.layers.append(nn.Conv2d(*config["final_conv"]))

    def forward(self, x):
        res = nn.Sequential(*self.layers)(x)
        return res