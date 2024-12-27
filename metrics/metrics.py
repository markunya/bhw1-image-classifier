import os
import random
import torch
import shutil
from utils.class_registry import ClassRegistry
from utils.data_utils import is_image_file

metrics_registry = ClassRegistry()

@metrics_registry.add_to_registry(name="accuracy")
class Accuracy:
    def __call__(self, predictions, labels):
        predicted_labels = predictions.argmax(dim=-1)
        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy
