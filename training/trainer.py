import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import classifiers_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from datasets.augmentations import mixes_registry
from training.losses import LossBuilder
from utils.data_utils import csv_to_list
from utils.data_utils import split_mapping
from tqdm import tqdm
from time import time

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
        self.setup_trainval_datasets()
        self.setup_dataloaders()

    def setup_inference(self):
        self.setup_classifier()
        self.setup_test_data()

    def setup_classifier(self):
        self.classifier = classifiers_registry[self.config['train']['classifier']](**self.config['train']['classifier_args']).to(self.device)

        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'])
            self.classifier.load_state_dict(checkpoint['classifier_state'])

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.classifier.parameters(), **self.config['train']['optimizer_args']
        )

        self.scheduler = schedulers_registry[self.config['train']['scheduler']](
            self.optimizer, **self.config['train']['scheduler_args']
        )

        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'])
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

    def setup_trainval_datasets(self):
        mapping = csv_to_list(os.path.join(self.config['data']['dataset_dir'], 'labels.csv'), key_column='Id', value_column='Category')
        if self.config['data']['val_size'] > 0:
            train_mapping, val_mapping = split_mapping(mapping, self.config['data']['val_size'], self.config['exp']['seed'])
        else:
            train_mapping = mapping
            val_mapping = []

        self.train_dataset = datasets_registry[self.config['data']['trainval_dataset']](self.config, train_mapping)
        self.val_dataset = datasets_registry[self.config['data']['trainval_dataset']](self.config, val_mapping)
        self.val_dataset.no_augs()

    def setup_test_data(self):
        self.test_dataset = datasets_registry[self.config['data']['test_dataset']](self.config)
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config['inference']['test_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
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
        num_epochs = self.config['train']['epochs']
        checkpoint_epoch = self.config['train']['checkpoint_epoch']

        for self.epoch in range(1, num_epochs + 1):
            running_loss = 0
            epoch_losses = {key: [] for key in self.loss_builder.losses.keys()}
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate at epoch {self.epoch}: {current_lr:.6f}")

            mixes = []
            for mix in self.config['mixes']:
                mixes.append(mixes_registry[mix]())

            with tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch}\{num_epochs}", unit="batch") as pbar:
                for batch in pbar:
                    if len(mixes) > 0:
                        batch = transforms.RandomChoice(mixes)(batch)
                    
                    losses_dict = self.train_step(batch)
                    
                    self.logger.update_losses(losses_dict)
                    for loss_name in epoch_losses.keys():
                        epoch_losses[loss_name].append(losses_dict[loss_name])
                    
                    running_loss = running_loss * 0.9 + losses_dict['total_loss'].item() * 0.1
                    pbar.set_postfix({"loss": running_loss})

            self.logger.log_train_losses(self.epoch)
            self.setup_train_dataloader()
            val_metrics_dict = self.validate()

            if self.config['train']['scheduler'] == 'reduce_on_plateau':
                self.scheduler.step(val_metrics_dict['val_' + self.config['train']['scheduler_metric']])
            else:
                self.scheduler.step()
                
            if val_metrics_dict is not None:
                self.logger.log_val_metrics(val_metrics_dict, epoch=self.epoch)

            if self.epoch % checkpoint_epoch == 0:
                self.save_checkpoint()

    def train_step(self, batch):
        self.to_train()
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
        run_name = self.config['exp']['run_name']
        torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'],
                                            f'checkpoint_{run_name}_{self.epoch}.pth'))

    @torch.no_grad()
    def validate(self):
        if len(self.val_dataset) == 0:
            return None
        
        self.to_eval()
        metrics_dict = {}

        for metric_name, _ in self.metrics:
            metrics_dict['train_' + metric_name] = 0
            metrics_dict['val_' + metric_name] = 0

        train_iter = iter(self.train_dataloader)
        val_iter = iter(self.val_dataloader)

        num_batches = min(len(self.train_dataloader), len(self.val_dataloader))

        for _ in range(num_batches):
            train_batch = next(train_iter)
            val_batch = next(val_iter)

            train_images = train_batch['images'].to(self.device)
            val_images = val_batch['images'].to(self.device)
            train_labels = train_batch['labels'].to(self.device)
            val_labels = val_batch['labels'].to(self.device)

            train_pred_logits = self.classifier(train_images)
            val_pred_logits = self.classifier(val_images)

            for metric_name, metric in self.metrics:
                metrics_dict['train_' + metric_name] += metric(train_pred_logits, train_labels) / num_batches
                metrics_dict['val_' + metric_name] += metric(val_pred_logits, val_labels) / num_batches

        print('Metrics: ', ", ".join(f"{key}={value}" for key, value in metrics_dict.items()))
        self.setup_val_dataloader()
        self.setup_train_dataloader()

        return metrics_dict



    @torch.no_grad()
    def inference(self, num_tta=10):
        self.to_eval()
        logits_dict = {}

        for _ in range(num_tta):
            for batch in self.test_dataloader:
                images = batch['images'].to(self.device)
                filenames = batch['filenames']

                pred_logits = self.classifier(images).cpu()

                for filename, logits in zip(filenames, pred_logits):
                    if filename not in logits_dict:
                        logits_dict[filename] = torch.zeros_like(logits)
                    logits_dict[filename] += logits / num_tta

        results = []
        for filename, avg_logits in logits_dict.items():
            predicted_label = avg_logits.argmax().item()
            results.append((filename, predicted_label))

        df = pd.DataFrame(results, columns=["Id", "Category"])
        output_file = "labels_test.csv"
        df.to_csv(output_file, index=False)

