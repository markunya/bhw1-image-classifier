## Описание
В данном репозитории представлена модель классификации изображений 40x40 на 200 классов. Подробнее с заданием можно ознакомиться на страничке Kaggle [соревнованния](https://www.kaggle.com/competitions/bhw-1-dl-2024-2025/overview).

## Note
Во всех экспериментах, если не сказано иначе, использовался:
- Размер батча: **64**
- Аугментации: **horizontal_flip + bcs**
- Функция потерь: **cross_entropy_loss**
- Шедулер: уменьшение learning rate в 10 раз на какой-то эпохе
- Модель: **resnet18**
- Random seed: **42**

Один из оптимизаторов:
- **Adam** с learning rate 0.0001
- **AdamW** с learning rate 0.0001
- **SGD** с learning rate 0.001 и momentum 0.9

---

## Аугментации

### Описание аугментаций:
- **horizontal_flip:** Зеркальное отражение по горизонтали — `torchvision.transforms.RandomHorizontalFlip`.
- **bcs:** `torchvision.transforms.ColorJitter` с нулевым параметром `hue`. Меняет яркость, контраст и насыщенность.
- **hue:** `torchvision.transforms.ColorJitter` с ненулевым параметром `hue`, который изменяет цветовой оттенок (например, лимон может стать оранжевым).
- **full_symmetry:** Отражение по вертикали или горизонтали, либо поворот на 90 градусов, создавая все возможные симметрии изображения.
- **affine:** `torchvision.transforms.RandomAffine`, применяющий повороты, сдвиги, перспективные трансформации.

### Выбор специальных аугментаций:
- **hue_special:** Аугментация `hue` применялась только для меток, где это не меняло отличительных признаков, например, для машин.
- **full_symmetry_special:** Аугментация применялась для симметричных объектов (например, божьих коровок), но не для асимметричных (например, хижин).

### Результаты экспериментов:
| Аугментации                                | Validation Accuracy | Train Accuracy |
|--------------------------------------------|---------------------|----------------|
| no augmentations                           | 0.339               | **0.706**      |
| horizontal_flip                            | 0.367               | 0.566          |
| **horizontal_flip + bcs**                  | **0.362**           | **0.522**      |
| horizontal_flip + bcs + hue                | 0.337               | 0.476          |
| horizontal_flip + bcs + full_symmetry      | 0.249               | 0.357          |
| horizontal_flip + bcs + hue_special        | 0.349               | 0.501          |
| horizontal_flip + bcs + full_symmetry_special | 0.349            | 0.495          |
| horizontal_flip + bcs + affine             | 0.201               | 0.251          |

---

## Миксы
Я проверил эффективность **CutMix** и **MixUp** на ResNet18 (128 каналов). Результаты:

| Миксы         | Validation Accuracy |
|---------------|---------------------|
| no mixes      | 0.36               |
| **cutmix**    | **0.39**           |
| mixup         | 0.335              |
| cutmix + mixup| 0.369              |
| cutmix 50%    | 0.357              |

CutMix показал лучший результат, даже если модель не видела "нормальные" изображения.

---

## Регуляризация
Модель по прежнему сильно переобучалась что недавало увеличить ее сложность. В статье про CutMix была табличка в которой показано, что CutMix + ShakeDrop работал лучше всего. Тогда я решил попробовать этот вид регуляризации. Он применяется не к класическому ResNet, а к его небольшой модификации - EraseReLU. В этой версии ReLU после сложения шортката и основного сверточного пути в классическом ResNet блоке убирается. С cutmix, shakedrop и параметром p\_drop\_max равным 0.25, удалось обучить модель до 0.43 на валидации и 0.59 на обучающей выборке. Как видно переобучение значительно снизилось и модель стала обладать лучшими обобщающими способностями. Пробовались различные значения параметра p\_drop\_max однако его увеличение приводило к значительному увеличению времени обучения. С параметром 0.5 модель обучалась на 15 эпохе дольше до такого же качества чем с параметром 0.25. Поэтому я остановился именно на 0.25.

---

## Оптимизаторы
Я пробовал обучать некоторые конфигурации ResNet с различными оптимизаторами, а именно Adam, AdamW и SGD+momentum. После подбора learning rate со всеми оптимизаторами удавалось добиться практически одних и тех же результатов. Поэтому я сделал вывод что оптимизаторы не сильно влиют на задачу и остановился на том, который считается наиболее продвинутым - AdamW.

---

## Конечное решение
Конечное решение:
- Модель: **ResNet18 (64 канала)**
- Аугментации: `horizontal_flip`, `bcs`, `crop` (RandomResizedCrop).
- Optimizer: AdamW
- Learning rate: 0.0001
- Epochs: 35
- Test-Time Augmentation (TTA): 10 на изображение, что повысило результат на тесте с **0.39 до 0.42**.

---

## Описание файла конфигурации
```json
{
    "exp": {
        // Название проекта. Для wandb
        "project_name": "dl_bhw_1",
        "device": "cuda",
        "seed": 42,
        // Использовать ли wandb
        "use_wandb": false
    },
    "data": {
        // Датасеты. Смотри datasets_registry
        "trainval_dataset": "simple_trainval_dataset",
        "test_dataset": "simple_test_dataset",
        // Путь к папке с датасетом
        "dataset_dir": "./",
        "train_batch_size": 64,
        "val_batch_size": 64,
        // Величина валидационной выборки
        "val_size": 0.04,
        "workers": 4
    },
    // Путь к чекпоинту откуда будут подгружены веса.
    "checkpoint_path": null,
    // Настройка инференса. Куда писать csv и с каким батчем это делать.
    "inference": {
        "output_dir": "./",
        "test_batch_size": 64
    },
    "train": {
        // Настройка модели классификации. Смотри classifiers_registry
        "classifier": "resnet",
        "classifier_args": {
            "num_blocks": [2, 2, 2, 2]
        },
        // Настройка оптимизатора. Смотри optimizers_registry 
        "optimizer": "adamW",
        "optimizer_args": {
            "lr": 0.0001
        },
        // Настройка шедулера. Смотри schedulers_registry
        "scheduler": "multi_step",
        "scheduler_metric": "accuracy",
        "scheduler_args": {
            "milestones": [30],
            "gamma": 0.1
        },
        // Список метрик для валидации. Определены в metrics_registry
        "val_metrics": [
            "accuracy"
        ],
        // Количество эпох для обучения
        "epochs": 35,
        // Шаг с которым сохраняются чекпоинты
        "checkpoint_epoch": 35,
        // Путь к папке в которую будут сохранены чекпоинты
        "checkpoints_dir": "./checkpoints"
    },
    // Словарь аугментаций. Определены в augmentations_registry.
    // Если значением выступает null то аугментация применяется ко всем меткам.
    // Если значение это список меток то аугментация будет применена только к этим меткам классов.
    "augmentations": {
        "horizontal_flip": null,
        "bcs": null,
        "crop": null
    },
    // Список используемых миксов.
    // Названия миксов определены в mixes_registry
    "mixes": ["cutmix"],
    "losses": {
        // Перечисление функций потерь и коэффициентов перед ними.
        // Итоговая функция потерь вычисляется как и взвешенная сумма.
        // Названия функций определены в losses_registry.
        "cross_entropy_loss": 1.0
    }
}
