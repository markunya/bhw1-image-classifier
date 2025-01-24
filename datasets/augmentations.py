from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
from utils.class_registry import ClassRegistry
import random
import torchvision.transforms.functional as F

augmentations_registry = ClassRegistry()
mixes_registry = ClassRegistry()

@augmentations_registry.add_to_registry(name='horizontal_flip')
class HorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

@augmentations_registry.add_to_registry(name='bcs')
class BCS(transforms.ColorJitter):
    def __init__(self, brightness=(0.6,1.2), contrast=0.3, saturation=0.4):
        super().__init__(brightness, contrast, saturation)

@augmentations_registry.add_to_registry(name='hue')
class Hue(transforms.ColorJitter):
    def __init__(self, hue=0.2):
        super().__init__(hue=hue)

@augmentations_registry.add_to_registry(name='crop')
class Crop(transforms.RandomResizedCrop):
    def __init__(self, size=40, scale=(0.64, 1.0), ratio=(0.9,1.1)):
        super().__init__(size=size, scale=scale, ratio=ratio)

@augmentations_registry.add_to_registry(name='perspective')
class Perspective(transforms.RandomPerspective):
    def __init__(self, distortion_scale=0.15, p=0.5, fill=(0.485, 0.456, 0.406)):
        super().__init__(distortion_scale=distortion_scale, p=p, fill=fill)

@augmentations_registry.add_to_registry(name='affine')
class Affine(transforms.RandomAffine):
    def __init__(self, degrees=30, translate=(0.1,0.1),
                scale=(0.9, 0.9), shear=(-10,10,-10,10),
                fill=(0.485, 0.456, 0.406)):
        super().__init__(degrees=degrees,translate=translate,scale=scale,shear=shear,fill=fill)

@augmentations_registry.add_to_registry(name='full_symmetry')
class FullSymmetry(nn.Module):
    def __init__(self):
        super(FullSymmetry, self).__init__()
        self.transform = transforms.RandomChoice(
            [
                transforms.RandomVerticalFlip(p=1.0),
                lambda img: F.rotate(img, random.choice([0,90,180,270]))
            ]
        )

    def forward(self, img):
        return self.transform(img)

@mixes_registry.add_to_registry(name='cutmix')    
class CutMix(v2.CutMix):
    def __init__(self, alpha=2.0, num_classes=200, labels_getter=lambda x: x['labels']):
        super().__init__(alpha=alpha, num_classes=num_classes, labels_getter=labels_getter)

@mixes_registry.add_to_registry(name='mixup')
class MixUp(v2.MixUp):
    def __init__(self, num_classes=200, labels_getter=lambda x: x['labels']):
        super().__init__(num_classes=num_classes, labels_getter=labels_getter)

@mixes_registry.add_to_registry(name='nomix')
class NoMix(nn.Module):
    def forward(self, batch):
        return batch
