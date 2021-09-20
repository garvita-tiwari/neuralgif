import argparse
from configs.config import load_config
# General config

from models import  check_weights
from models import load_data
import ipdb

def train(opt):
    data_load =getattr(load_data, opt['experiment']['data_name'])

    train_dataset = data_load('train', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'])
    val_dataset = data_load('test', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'])

    trainer = getattr(check_weights, opt['experiment']['type'])
    trainer = trainer( train_dataset=train_dataset, val_dataset=val_dataset, opt=opt)
    trainer.train_model(opt['train']['max_epoch'], eval=opt['train']['eval'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NeuralGIF.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config('configs/smpl.yaml')

    train(opt)