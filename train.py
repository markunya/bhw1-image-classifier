import json
import argparse
from utils.model_utils import setup_seed
from training.trainer import Trainer

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

if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    config['exp']['run_name'] = args.run_name

    trainer = Trainer(config)

    trainer.setup()
    trainer.training_loop()