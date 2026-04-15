from src.modules.rectified_flow import *
from src.utils.data_utils import *
from src.utils.utils import *
import torch
import argparse
import yaml
import shutil
from tqdm import tqdm
from datetime import datetime

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
            '--mode',
            choices=['train', 'debug', 'overfit'],
            default='train',
            help='mode of running train script, default: train')

    parser.add_argument(
            '--debug', type=bool,
            default=False,
            help='turn on debug mode')

    parser.add_argument(
            '--epochs', type=int,
            help='num of epochs')

    parser.add_argument(
            '--batch_size', type=int)

    parser.add_argument(
            '--num_training', type=int)

    return parser.parse_args()


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


def train_rectified_flow_model(model, optimizer, criterion, data_loader, config, debug=False):
    epochs = config['train']['process']['epochs']
    device = config['device']
    save_model_every_n_epochs = config['checkpoint']['model_save_every_n_epochs']

    save_img_every_n_epochs = config['checkpoint']['img_save_every_n_epochs']
    noise_for_imgs = None
    if save_img_every_n_epochs:
        noise_for_imgs = torch.randn(size=(config['checkpoint']['img_B'], model.in_channels, model.img_size, model.img_size))

    # savedir and save config
    if not(debug):
        run_id_part_1 = datetime.now().strftime("%Y-%m-%d")
        run_id_part_2 = datetime.now().strftime("%H-%M-%S") + f"ViT-{model.n_heads}_{model.patch_size}_ep_{epochs}"
        run_id = os.path.join(run_id_part_1, run_id_part_2)
        save_dir = os.path.join(config['checkpoint']['saveroot'], run_id)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = config['checkpoint']['saveroot']
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.makedirs(save_dir)

    if save_img_every_n_epochs:
        save_img_path = os.path.join(save_dir, "images")
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    
    with open(os.path.join(save_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    model = model.train()
    avg_loss = None

    for epoch in range(epochs):
        total_loss, total_num = 0.0, 0.0
        for x in tqdm(data_loader):
            B, C, N, N = x.shape
            x = x.float().to(device)
            noise = torch.randn(size=(B, C, N, N)).to(device)
            t = torch.rand(size=(B, 1, 1, 1)).to(device)
            noised_image = (1 - t) * noise + t * x
            target = x - noise # target vector field
            pred = model(noised_image, t.reshape(B, 1, 1))
            
            optimizer.zero_grad()         
            batch_loss = criterion(pred, target)
            batch_loss.backward()
            optimizer.step()
            
            total_num += B
            total_loss += batch_loss.item() * B

        avg_loss = total_loss / total_num

        if (epoch + 1) == epochs or (save_model_every_n_epochs and (epoch + 1) % save_model_every_n_epochs == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "noise_for_imgs": noise_for_imgs,
                          "epoch": epoch,
                          "best_loss": avg_loss,
            }

            torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

        if (epoch + 1) == epochs or (save_img_every_n_epochs and (epoch + 1) % save_img_every_n_epochs == 0):
            save_img(model=model,
                     B=config['checkpoint']['img_B'],
                     T=config['checkpoint']['img_T'], 
                     path=save_img_path,
                     epoch=epoch + 1,
                     device=device,
                     noise_for_img=noise_for_imgs,
                     with_process=True)

        print(f"epoch {epoch + 1}/{epochs}. Loss: {avg_loss:.4f}")
    return avg_loss


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


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    model = init_model(config)
    optimizer = init_optimizer(model, config)   
    loss_instance = torch.nn.MSELoss()
    data_loader = init_data_loader(config)
    
    epochs = config['train']['process']['epochs']
    device = config['device']

    debug_mode = args.mode == 'debug'

    train_rectified_flow_model(model=model, optimizer=optimizer, criterion=loss_instance, data_loader=data_loader, config=config, debug=debug_mode)
