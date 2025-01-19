import json
from utils.model_utils import setup_seed
from training.trainer import Trainer
from utils.data_utils import read_json_file

if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])

    trainer = Trainer(config)

    trainer.setup_inference()
    trainer.inference()