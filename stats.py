import csv
from collections import defaultdict, Counter
import torch
from tqdm import tqdm
from utils.model_utils import setup_seed
from training.trainer import Trainer
from utils.data_utils import read_json_file


def generate_stats(dataloader, classifier, output_csv_path='stats.csv'):
    classifier.eval()

    label_counts = Counter()  
    correct_counts = Counter()
    confusion_matrix = defaultdict(Counter)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing dataset")):
            inputs = batch['images'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = classifier(inputs)
            predictions = torch.argmax(outputs, dim=1)

            for label, prediction in zip(labels, predictions):
                label_counts[label.item()] += 1
                if label == prediction:
                    correct_counts[label.item()] += 1
                else:
                    confusion_matrix[label.item()][prediction.item()] += 1

    total_correct = sum(correct_counts.values())
    total_samples = sum(label_counts.values())
    overall_accuracy = total_correct / total_samples

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Label', 'Total Count', 'Correct Count', 'Accuracy', 
            'Most Confused With', 'Confusion Count', 'Overall Accuracy'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for label in label_counts:
            total_count = label_counts[label]
            correct_count = correct_counts[label]
            accuracy = correct_count / total_count if total_count > 0 else 0

            if confusion_matrix[label]:
                most_confused_label, confusion_count = confusion_matrix[label].most_common(1)[0]
            else:
                most_confused_label, confusion_count = None, 0

            writer.writerow({
                'Label': label,
                'Total Count': total_count,
                'Correct Count': correct_count,
                'Accuracy': accuracy,
                'Most Confused With': most_confused_label,
                'Confusion Count': confusion_count,
                'Overall Accuracy': overall_accuracy
            })

    print(f"Stats saved to {output_csv_path}")


if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])

    trainer = Trainer(config)

    trainer.setup_classifier()
    trainer.setup_trainval_datasets()
    trainer.setup_val_dataloader()

    dataloader = trainer.val_dataloader
    classifier = trainer.classifier

    generate_stats(dataloader, classifier)
    