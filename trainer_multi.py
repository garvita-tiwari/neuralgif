import argparse
from configs.config import load_config
# General config

from models import  train_multishape
from models import load_data
import shutil


def train(opt,config_file):

    data_load =getattr(load_data, opt['experiment']['data_name'])

    train_dataset = data_load('train', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'], shape=True)
    val_dataset = data_load('test', data_path=opt['data']['data_dir'], split_file=opt['data']['split_file'], batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'], shape=True)

    trainer = getattr(train_multishape, opt['experiment']['type'])
    trainer = trainer( train_dataset=train_dataset, val_dataset=val_dataset, opt=opt)


    copy_config = '{}/{}/{}'.format(opt['experiment']['root_dir'], trainer.exp_name, 'config.yaml')
    shutil.copyfile(config_file,copy_config )

    trainer.train_model(opt['train']['max_epoch'], eval=opt['train']['eval'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NeuralGIF.'
    )
    parser.add_argument('--config', '-c', default='configs/smpl_multi.yaml', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config)
    #save the config file

    train(opt, args.config)