from rectified_flow import *
from data_utils import *
from utils import *
import torch
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--config', type=str,
            help='name of config file to load',
            default='configs/default_config.yaml')

    parser.add_argument(
            '--device', type=str,
            default='cuda:0',
            help='which device to use on local machine')
    
    parser.add_argument(
            '--debug', type=bool,
            default=False,
            help='turn on debug mode')

    parser.add_argument(
            '--epochs', type=int,
            help='num of epochs')

    parser.add_argument(
            '--batch_size', type=int)

    return parser.parse_args()

def load_config(args):
    if args.debug:
        args.config = 'configs/debug_config.yaml'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['device'] = args.device
    if config['device'] == 'cuda:0':
        assert config['device'] == 'cuda:0' and torch.cuda.is_available(), f"cuda is not available"

    if args.epochs is not None:
        config['train']['process']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['train']['data']['batch_size'] = args.batch_size
    
    return config

def init_model(config):
    model_config = config['model']
    device = config['device']

    img_size = model_config['img_size']
    in_channels = model_config['in_channels']
    patch_size = model_config['patch_size']
    emb_dim = model_config['emb_dim']
    ffn_dim_ratio = model_config['ffn_dim_ratio']
    n_heads = model_config['n_heads']
    num_layers = model_config['num_layers']

    model = RectifiedFlowViT(img_size=img_size,
                             in_channels=in_channels,
                             patch_size=patch_size,
                             emb_dim=emb_dim,
                             ffn_dim_ratio=ffn_dim_ratio,
                             n_heads=n_heads,
                             num_layers=num_layers).to(device)

    return model

def init_data_loader(config):
    data_config = config['train']['data']

    num_training = data_config['num_training']
    num_validation = data_config['num_validation']

    batch_size = data_config['batch_size']
    drop_last = data_config['drop_last']

    data_dict = get_CIFAR10_data(num_training, num_validation)

    data_loader = torch.utils.data.DataLoader(data_dict['X_train'], batch_size=batch_size, drop_last=drop_last) # for overfit do this with X_train: expand(batch_size, -1, -1, -1)

    return data_loader

def init_optimizer(model, config):
    optimizer_config = config['train']['optimizer']
    
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    
    return optimizer
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    model = init_model(config)
    optimizer = init_optimizer(model, config)   
    loss_instance = torch.nn.MSELoss()
    data_loader = init_data_loader(config)
    
    epochs = config['train']['process']['epochs']
    device = config['device']

    train_rectified_flow_model(model=model, optimizer=optimizer, criterion=loss_instance, data_loader=data_loader, config=config)
