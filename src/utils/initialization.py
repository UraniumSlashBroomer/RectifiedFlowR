import torch
from .data_utils import get_CIFAR10_data
from ..modules.rectified_flow import RectifiedFlowViT, EMAModel
from .utils import get_config_checkpoint_path
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


def init_ema(model, config):
    try:
        decay = config['model']['ema_model']['decay']
    except KeyError:
        decay = 0.999

    ema_model = EMAModel(model, decay=decay)
    return ema_model


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


def init_scheduler(optimizer, config):
    try:
        type = config['train']['scheduler']['type']
    except KeyError:
        type = None
    
    scheduler = None
    data_config = config['train']['data']
    total_batches = data_config['num_training'] // data_config['batch_size'] * config['train']['process']['epochs']

    if type == 'CosineAnnealingLR':
        eta_min = config['train']['scheduler']['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=total_batches,
                                                               eta_min=eta_min)
    else:
        print(f"no scheduler in config")
    
    return scheduler
    

def load_checkpoint(experiment_path):
    config_path, checkpoint_path = get_config_checkpoint_path(experiment_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = init_model(config)
    ema_model = init_ema(model, config)
    data_loader = init_data_loader(config)
    optimizer = init_optimizer(model, config)
    scheduler = init_scheduler(optimizer, config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    avg_loss = checkpoint['avg_loss']
    noise_for_imgs = checkpoint['noise_for_imgs']

    return model, ema_model, data_loader, optimizer, scheduler, epoch, avg_loss, noise_for_imgs, config


def load_config(args):
    if args.mode == 'debug':
        args.config = 'configs/debug_config.yaml'
    elif args.mode == 'overfit':
        args.config = 'configs/overfit_config.yaml'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['device'] = args.device
    if config['device'] == 'cuda:0':
        assert config['device'] == 'cuda:0' and torch.cuda.is_available(), f"cuda is not available"

    if args.epochs is not None:
        config['train']['process']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['train']['data']['batch_size'] = args.batch_size

    if args.num_training is not None:
        config['train']['data']['num_training'] = args.num_training
    
    return config
