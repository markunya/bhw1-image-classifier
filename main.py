import json
from utils.model_utils import setup_seed
from training.trainer import Trainer
from utils.data_utils import read_json_file

if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])
    config['exp']['use_wandb'] = False
    config['exp']['run_name'] = 'inference'
    config['data']['val_size'] = 0.0

    trainer = Trainer(config)

    trainer.setup()
    trainer.training_loop()

    trainer.setup_test_data()
    trainer.inference()
