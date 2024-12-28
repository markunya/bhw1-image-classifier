from torchvision import transforms
from utils.class_registry import ClassRegistry

augmentations_registry = ClassRegistry()

@augmentations_registry.add_to_registry(name='horizontal_flip')
class HorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

@augmentations_registry.add_to_registry(name='rotation')
class Rotation(transforms.RandomRotation):
    def __init__(self, degrees=30):
        super().__init__(degrees)

@augmentations_registry.add_to_registry(name='color_jitter')
class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        super().__init__(brightness, contrast, saturation, hue)
