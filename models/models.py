import torch
from torch import nn
from utils.class_registry import ClassRegistry
from utils.model_utils import weights_init

classifiers_registry = ClassRegistry()

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=(-1, 1)):
        if not training:
            ctx.save_for_backward(None)
            return x * (1 - p_drop)

        gate = torch.bernoulli(torch.tensor(1 - p_drop, device=x.device)).item()

        if gate == 0:
            alpha = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*alpha_range)
            alpha = alpha.expand_as(x)
            ctx.save_for_backward(torch.tensor(0), alpha)
            return alpha * x
        else:
            ctx.save_for_backward(torch.tensor(1))
            return x

    @staticmethod
    def backward(ctx, grad_output):
        gate, _ = ctx.saved_tensors
        if gate.item() == 0:
            beta = torch.empty(grad_output.size(0), 1, 1, 1, device=grad_output.device).uniform_(0, 1)
            beta = beta.expand_as(grad_output)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=(-1, 1)):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)
    
class BaseResnetBlock(nn.Module):
    def __init__(self, block, shortcut, shakedrop=False, p_drop=0.5):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.relu = nn.ReLU()
        self.shakedrop = ShakeDrop(p_drop=p_drop) if shakedrop else None

    def forward(self, x):
        residual = self.block(x) + self.shortcut(x)
        if self.shakedrop:
            residual = self.shakedrop(residual)
        return self.relu(residual)

class ResnetBlock(BaseResnetBlock):
    def __init__(self, channels, downsample=False, shakedrop=False, p_drop=0.5):
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
                start_kernel=3, shakedrop=False, p_drop_max=0.5):
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
        total_blocks = sum(num_blocks)
        current_block_index = 1
        def getpdrop():
            nonlocal current_block_index
            nonlocal total_blocks
            p_drop = 1 - current_block_index * (1 - p_drop_max) / total_blocks
            current_block_index += 1
            return p_drop
        
        for i in range(len(num_blocks)):
            block_of_blocks = [None] * num_blocks[i]
            for j in range(num_blocks[i] - 1):
                block_of_blocks[j] = ResnetBlock(current_channels, shakedrop=shakedrop, p_drop=getpdrop())
            block_of_blocks[-1] = ResnetBlock(current_channels, downsample=True, shakedrop=shakedrop, p_drop=getpdrop())
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
