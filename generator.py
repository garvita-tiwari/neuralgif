import argparse
from configs.config import load_config
# General config

from models import  generate_shape
from models import load_data
import ipdb
from generation_iterator import gen_iterator

def train(opt):
    data_load =getattr(load_data, opt['experiment']['data_name'])

    train_dataset = data_load('train', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=1, num_workers=opt['train']['num_worker'])
    val_dataset = data_load('test', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=1, num_workers=opt['train']['num_worker'])
    checkpoint = 14350
    out_path = '{}/{}/{}'.format(opt['experiment']['root_dir'], opt['experiment']['exp_name'], checkpoint)
    gen = getattr(generate_shape, opt['experiment']['type'])
    gen = gen( opt=opt, checkpoint=checkpoint)
    gen_iterator(out_path, train_dataset, gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NeuralGIF.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config('configs/smpl.yaml')

    train(opt)