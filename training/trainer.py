import os
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import classifiers_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.losses import LossBuilder
from utils.data_utils import csv_to_list
from utils.data_utils import split_mapping
from tqdm import tqdm

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['exp']['device']
        self.epoch = None

    def setup(self):
        self.setup_classifier()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_metrics()
        self.setup_logger()
        self.setup_datasets()
        self.setup_dataloaders()

    def setup_inference(self):
        self.setup_classifier()
        self.setup_metrics()
        self.setup_logger()
        self.setup_datasets()
        self.setup_dataloaders()

    def setup_classifier(self):
        self.classifier = classifiers_registry[self.config['train']['classifier']](**self.config['train']['classifier_args']).to(self.device)

        if self.config['train']['checkpoint_path']:
            checkpoint = torch.load(self.config['train']['checkpoint_path'])
            self.classifier.load_state_dict(checkpoint['classifier_state'])

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.classifier.parameters(), **self.config['train']['optimizer_args']
        )

        if self.config['train']['checkpoint_path']:
            checkpoint = torch.load(self.config['train']['checkpoint_path'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def setup_losses(self):
        self.loss_builder = LossBuilder(self.config)

    def to_train(self):
        self.classifier.train()

    def to_eval(self):
        self.classifier.eval()

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config['train']['val_metrics']:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_datasets(self):
        mapping = csv_to_list(os.path.join(self.config['data']['dataset_dir'], 'labels.csv'), key_column='Id', value_column='Category')
        train_mapping, val_mapping = split_mapping(mapping, self.config['data']['val_size'], self.config['exp']['seed'])

        self.train_dataset = datasets_registry[self.config['data']['dataset']](
            root = self.config['data']['dataset_dir'],
            mapping = train_mapping,
            transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
            )

        self.val_dataset = datasets_registry[self.config['data']['dataset']](
            root = self.config['data']['dataset_dir'],
            mapping = val_mapping,
            transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
        )
    
    def setup_train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            shuffle=True,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
        )

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['val_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
        )

    def setup_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()

    def training_loop(self):
        self.to_train()

        num_epochs = self.config['train']['epochs']
        checkpoint_epoch = self.config['train']['checkpoint_epoch']

        for self.epoch in range(1, num_epochs + 1):
            epoch_losses = {key: [] for key in self.loss_builder.losses.keys()}
            with tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch}\{num_epochs}", unit="batch") as pbar:
                for batch in pbar:
                    losses_dict = self.train_step(batch)
                    self.logger.update_losses(losses_dict)
                    for loss_name in epoch_losses.keys():
                        epoch_losses[loss_name].append(losses_dict[loss_name])
                    pbar.set_postfix({"loss": losses_dict['total_loss'].item()})
            
            self.logger.log_train_losses(self.epoch)

            val_metrics_dict = self.validate()
            self.logger.log_val_metrics(val_metrics_dict, epoch=self.epoch)

            if self.epoch % checkpoint_epoch == 0:
                self.save_checkpoint()

            self.setup_train_dataloader()

    def train_step(self, batch):
        self.optimizer.zero_grad()

        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)

        pred_logits = self.classifier(images)
        loss_dict = self.loss_builder.calculate_loss(pred_logits, labels)
        loss_dict['total_loss'].backward()

        self.optimizer.step()
        return loss_dict

    def save_checkpoint(self):
        checkpoint = {
            'classifier_state': self.classifier.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'], f'checkpoint_{self.epoch}.pth'))

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        metrics_dict = {}
        for metric_name, _ in self.metrics:
            metrics_dict[metric_name] = 0

        for batch in self.val_dataloader:
            for metric_name, metric in self.metrics:
                images = batch['images'].to(self.device)    
                labels = batch['labels'].to(self.device)

                pred_logits = self.classifier(images)
                metrics_dict[metric_name] += metric(pred_logits, labels) / len(self.val_dataloader)
        
        print('Validation metrics: ', ", ".join(f"{key}={value}" for key, value in metrics_dict.items()))
        
        self.setup_val_dataloader()
        return metrics_dict

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        for batch in self.val_dataloader:
            images = batch['images'].to(self.device)
            pred_logits = self.classifier(images)
            print(pred_logits)
