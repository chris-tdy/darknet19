import torch
import torch.nn as nn
import torch.nn.functional as F


class Darknet19(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()

        # 定义卷积块
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.layer1 = nn.Sequential(
            conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 64),
            conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 128),
            conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            conv_block(512, 1024),
            conv_block(1024, 512),
            conv_block(512, 1024),
            conv_block(1024, 512),
            conv_block(512, 1024)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = Darknet19()
input_data = torch.randn(1, 3, 224, 224)  # 假设输入是一张224x224大小的RGB图像
output = model(input_data)  # 自动调用 forward 方法
print(model)
