import argparse
from configs.config import load_config
# General config

from models import  generate_multi
from models import load_data
import ipdb
from generation_iterator import gen_iterator

def train(opt):
    data_load =getattr(load_data, opt['experiment']['data_name'])

    train_dataset = data_load('train', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=1, num_workers=opt['train']['num_worker'], shape=True)
    val_dataset = data_load('test', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=1, num_workers=opt['train']['num_worker'],shape=True)
    checkpoint = 19999
    resolution = 256
    gen = getattr(generate_multi, opt['experiment']['type'])
    gen = gen( opt=opt, checkpoint=checkpoint, resolution=resolution)
    out_path = '{}/{}/{}_{}'.format(opt['experiment']['root_dir'], gen.exp_name, checkpoint,resolution)

    gen_iterator(out_path, val_dataset, gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate NeuralGIF.'
    )
    parser.add_argument('--config', '-c', default='configs/smpl_multi.yaml', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config)

    train(opt)