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
    positional_encoding = model_config['positional_encoding']
    p_pos_encoding_dropout = model_config['p_pos_encoding_dropout']
    p_encoder_dropout = model_config['p_encoder_dropout']

    model = RectifiedFlowViT(img_size=img_size,
                             in_channels=in_channels,
                             patch_size=patch_size,
                             emb_dim=emb_dim,
                             ffn_dim_ratio=ffn_dim_ratio,
                             n_heads=n_heads,
                             num_layers=num_layers,
                             positional_encoding=positional_encoding,
                             p_pos_encoding_dropout=p_pos_encoding_dropout,
                             p_encoder_dropout=p_encoder_dropout).to(device)
    
    print(f"total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
    
    optimizer = torch.optim.AdamW(lr=lr,
                                  params=model.parameters(),
                                  weight_decay=weight_decay)
    
    return optimizer


def init_scheduler(optimizer, config):
    try:
        type = config['train']['scheduler']['type']
    except KeyError:
        type = None
    
    try:
        warmup_type = config['train']['scheduler']['warmup']['type']
        warmup_epochs = config['train']['scheduler']['warmup']['epochs']
        warmup_start_factor = config['train']['scheduler']['warmup']['start_factor']
        assert warmup_epochs < config['train']['process']['epochs'], f"warmup epochs should be less than total epochs"
    except KeyError:
        warmup_type = None
        warmup_epochs = 0

    scheduler = None
    warmup_scheduler = None
    constant_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    result_scheduler = None

    data_config = config['train']['data']
    if data_config['num_training'] < data_config['batch_size']:
        batches_in_epoch = 1
    else:
        batches_in_epoch = data_config['num_training'] // data_config['batch_size']

    total_warmup_batches = batches_in_epoch * warmup_epochs
    total_scheduler_batches =  batches_in_epoch * config['train']['process']['epochs'] - total_warmup_batches

    if type == 'CosineAnnealingLR':
        eta_min = config['train']['scheduler']['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=total_scheduler_batches,
                                                               eta_min=eta_min)
    else:
        print(f"no scheduler in config")

    if warmup_type == 'linear':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                          start_factor=warmup_start_factor,
                                          total_iters=total_warmup_batches)
    else:
        print(f"no warmup in config")
    
    if scheduler is not None and warmup_scheduler is not None:
        result_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler, constant_scheduler],
                milestones=[total_warmup_batches, total_warmup_batches + total_scheduler_batches + 1])
    elif scheduler is not None:
        result_scheduler = torch.optim.lr_scheduler(optimizer,
                                                    schedulers=[scheduler, constant_scheduler],
                                                    milestones=[total_scheduler_batches + 1])
        
    return result_scheduler
    

def load_train_checkpoint(experiment_path, args):
    config_path, checkpoint_path = get_config_checkpoint_path(experiment_path)

    config = load_config(args, config_path)
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

def load_eval_checkpoint(experiment_path, args):
    config_path, checkpoint_path = get_config_checkpoint_path(experiment_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config['device'] = args.device

    model = init_model(config)
    ema_model = init_ema(model, config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    return ema_model, config


def load_config(args, config_path=None):
    if args.mode == 'debug':
        args.config = 'configs/debug_config.yaml'
    elif args.mode == 'overfit':
        args.config = 'configs/overfit_config.yaml'
    elif args.mode == 'train_c':
        args.config = config_path

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
    
    if args.decay is not None:
        config['model']['ema_model']['decay'] = args.decay

    if args.warmup_epochs is not None:
        config['train']['scheduler']['warmup']['epochs'] = args.warmup_epochs

    return config
