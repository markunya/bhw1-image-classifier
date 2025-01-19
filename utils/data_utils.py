import os
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import defaultdict
import pandas as pd
import json

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
    return list(zip(df[key_column], df[value_column]))

def split_mapping(mapping, val_size, random_state):
    grouped = defaultdict(list)
    for item, label in mapping:
        grouped[label].append((item, label))
    
    train_mapping = []
    val_mapping = []
    
    for label, items in grouped.items():
        train_items, val_items = train_test_split(
            items,
            test_size=val_size,
            random_state=random_state
        )
        train_mapping.extend(train_items)
        val_mapping.extend(val_items)
    
    return train_mapping, val_mapping

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка при декодировании JSON из файла '{file_path}'.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
