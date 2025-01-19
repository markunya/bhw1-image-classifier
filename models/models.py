import torch
from torch import nn
from utils.class_registry import ClassRegistry
from utils.model_utils import weights_init

classifiers_registry = ClassRegistry()
    
class BaseResnetBlock(nn.Module):
    def __init__(self, block, shortcut):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class ResnetBlock(BaseResnetBlock):
    def __init__(self, channels, downsample=False):
        stride = 1
        expansion = 1
        if downsample:
            stride = 2
            expansion = 2
        BaseResnetBlock.__init__(
            self,
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels*expansion, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(channels*expansion),
            ),
            shortcut=nn.Conv2d(channels, channels*expansion, kernel_size=1, stride=stride)
        )
        

@classifiers_registry.add_to_registry(name='resnet')
class ResNet0(nn.Module):
    def __init__(self, img_size=40, num_blocks=None,
                num_classes=200, start_channels=64,
                start_kernel=3):
        super().__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        blocks = [
                nn.Conv2d(3, start_channels, kernel_size=start_kernel, padding=start_kernel//2),
                nn.BatchNorm2d(start_channels),
                nn.ReLU(),
        ]

        reduced_size = img_size
        current_channels = start_channels
        for i in range(len(num_blocks)):
            block_of_blocks = [None] * num_blocks[i]
            for j in range(num_blocks[i] - 1):
                block_of_blocks[j] = ResnetBlock(current_channels)
            block_of_blocks[-1] = ResnetBlock(current_channels, downsample=True)
            blocks += block_of_blocks
            current_channels *= 2
            reduced_size = (reduced_size + 1) // 2

        blocks += [
            nn.Conv2d(current_channels, current_channels*reduced_size, kernel_size=2, padding=1),
            nn.AvgPool2d(kernel_size=reduced_size),
            nn.Flatten(),
            nn.Linear(current_channels * reduced_size, num_classes)
        ]

        self.model = nn.Sequential(*blocks)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)
