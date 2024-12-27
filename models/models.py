import torch
from torch import nn
from utils.class_registry import ClassRegistry
from utils.model_utils import weights_init

classifiers_registry = ClassRegistry()

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

@classifiers_registry.add_to_registry(name='v_0_0')
class Classifier_V_0_0(nn.Module):
    def __init__(self, img_size, num_blocks, num_classes):
        super().__init__()
        assert img_size % (2**num_blocks) == 0, "Image size must be divisible by 2^num_blocks"

        blocks = [
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, 4, kernel_size=3),
                ResnetBlock(4, 4)
            )
        ]
        current_channels = 4
        for _ in range(num_blocks):
            blocks += [
                nn.MaxPool2d(kernel_size=2),
                ResnetBlock(current_channels, current_channels * 2)
            ]
            current_channels *= 2

        reduced_size = img_size // (2**num_blocks)
        blocks += [
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(current_channels * reduced_size * reduced_size, num_classes)
        ]

        self.model = nn.Sequential(*blocks)

        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)
