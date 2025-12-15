import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODEUR
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # DECODEUR
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = nn.Conv2d(64, 64, 3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.up1(x)
        x = F.relu(self.deconv1(x))

        x = self.up2(x)
        x = F.relu(self.deconv2(x))

        x = torch.sigmoid(self.out(x))
        return x
