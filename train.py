import argparse
from utils.model_utils import setup_seed
from training.trainer import Trainer
from utils.data_utils import read_json_file

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