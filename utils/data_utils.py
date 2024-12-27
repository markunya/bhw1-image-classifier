import os
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_images_from_dir(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for filename in os.listdir(dir):
        if is_image_file(filename):
            images.append(filename)
    return images

def csv_to_list(file_path, key_column, value_column):
    df = pd.read_csv(file_path)
    image_label_dict = list(zip(df[key_column], df[value_column]))
    return image_label_dict

def split_mapping(mapping, val_size, random_state):
    train_mapping, val_mapping = train_test_split(
        mapping,
        test_size=val_size,
        random_state=random_state
    )
    return train_mapping, val_mapping

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))
