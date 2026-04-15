import torch
from .data_utils import get_CIFAR10_data
from ..modules.rectified_flow import RectifiedFlowViT
import yaml


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
    
    if num_training < batch_size:
        data_loader = torch.utils.data.DataLoader(data_dict['X_train'].repeat(batch_size // num_training, 1, 1, 1), batch_size=batch_size)
    else:
        data_loader = torch.utils.data.DataLoader(data_dict['X_train'], batch_size=batch_size, drop_last=drop_last)

    return data_loader


def init_optimizer(model, config):
    optimizer_config = config['train']['optimizer']
    
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    
    return optimizer


def load_checkpoint(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = init_model(config)
    data_loader = init_data_loader(config)
    optimizer = init_optimizer(model, config)
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    return model, data_loader, optimizer, epoch, best_loss, config
