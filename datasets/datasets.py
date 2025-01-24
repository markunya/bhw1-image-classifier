import os
from torchvision import transforms
from torchvision.io import read_image
from utils.class_registry import ClassRegistry
from utils.data_utils import get_images_from_dir
from datasets.augmentations import augmentations_registry
from torch.utils.data import Dataset

datasets_registry = ClassRegistry()
ts_tail = [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]

unnormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

class AugmentationsBuilder:
    def __init__(self, aug_mapping):
        self.aug_mapping = {key:None if value is None else set(value)
                            for key, value in aug_mapping.items()}
        self.label_to_transform = {}

    def apply_augmentation(self, img, label):
        if label in self.label_to_transform:
            return self.label_to_transform[label](img)
        
        ts = []
        for aug_name, labels in self.aug_mapping.items():
            if labels is None or label in labels:
                ts.append(augmentations_registry[aug_name]())
        ts += ts_tail

        transform = transforms.Compose(ts)
        self.label_to_transform[label] = transform
        return transform(img)

    def set_transforms(self, ts):
        self.transforms = transforms.Compose(ts+ts_tail)

    def no_augs(self):
        self.set_transforms(ts=[])

@datasets_registry.add_to_registry(name='simple_trainval_dataset')
class TrainValDataset(Dataset, AugmentationsBuilder):
    def __init__(self, config, mapping):
        AugmentationsBuilder.__init__(self, config['augmentations'])
        self.trainval_dir = os.path.join(config['data']['dataset_dir'], 'trainval')
        self.mapping = mapping
        self.transforms = None

    def __getitem__(self, ind):
        label = self.mapping[ind][1]
        path = os.path.join(self.trainval_dir, self.mapping[ind][0])
        image = read_image(path).float() / 255.0

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = self.apply_augmentation(image, label)

        return {'images': image, 'labels': label}

    def __len__(self):
        return len(self.mapping)
    
@datasets_registry.add_to_registry(name='simple_test_dataset')
class TestDataset(Dataset):
    def __init__(self, config):
        augs_list = [augmentations_registry[key]()
                    for key, value in config['augmentations'].items() if value is None]
        self.augs = transforms.Compose(augs_list+ts_tail)
        
        self.test_dir = os.path.join(config['data']['dataset_dir'], 'test')
        self.images = get_images_from_dir(self.test_dir)
        
    def __getitem__(self, ind):
        path = os.path.join(self.test_dir,  self.images[ind])
        image = read_image(path).float() / 255.0
        image = self.augs(image)
        return {'images': image, 'filenames': self.images[ind]}

    def __len__(self):
        return len(self.images)
