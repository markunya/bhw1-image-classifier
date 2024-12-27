import os
from PIL import Image
from utils.class_registry import ClassRegistry
from torch.utils.data import Dataset

datasets_registry = ClassRegistry()

@datasets_registry.add_to_registry(name='given')
class SimpleDataset(Dataset):
    def __init__(self, root, mapping, transforms=None):
        self.root = root
        self.transforms = transforms
        self.mapping = mapping

    def __getitem__(self, ind):
        label = self.mapping[ind][1]
        path = os.path.join(self.root, 'trainval', self.mapping[ind][0])
        image = Image.open(path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return {'images': image, 'labels': label}

    def __len__(self):
        return len(self.mapping)
