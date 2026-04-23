from src.modules.rectified_flow import *
from src.utils.data_utils import *
from src.utils.utils import *
from src.utils.initialization import *
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
            choices=['train', 'debug', 'overfit', 'train_c'],
            default='train',
            help='mode of running train script, default: train')

    parser.add_argument(
            '--epochs', type=int,
            help='num of epochs')

    parser.add_argument(
            '--batch_size', type=int)

    parser.add_argument(
            '--num_training', type=int)

    parser.add_argument(
            '--experiment', type=str)

    parser.add_argument(
            '--decay', type=float)

    return parser.parse_args()


def prepare_saving(model, config, start_epoch, debug=False):
    epochs = config['train']['process']['epochs']
    save_model_every_n_epochs = config['checkpoint']['model_save_every_n_epochs']
    save_img_every_n_epochs = config['checkpoint']['img_save_every_n_epochs']
    save_img_path = None

    # savedir and save config
    if not(debug):
        run_id_part_1 = datetime.now().strftime("%Y-%m-%d")
        run_id_part_2 = datetime.now().strftime("%H-%M-%S") + f"ViT-{model.n_heads}_{model.patch_size}_ep_{start_epoch}_{epochs}"
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

    return save_model_every_n_epochs, save_img_every_n_epochs, save_img_path, save_dir


def save_checkpoint(epoch, epochs, save_model_every_n_epochs, model, ema_model, optimizer, scheduler, noise_for_imgs, avg_loss, save_dir):
    if epoch == epochs or (save_model_every_n_epochs and epoch % save_model_every_n_epochs == 0):
        checkpoint = {"model_state_dict": model.state_dict(),
                      "ema_model_state_dict": ema_model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                      "noise_for_imgs": noise_for_imgs,
                      "epoch": epoch,
                      "avg_loss": avg_loss,
        }

        torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))


def train_rectified_flow_model(model, ema_model, scheduler, optimizer, criterion, data_loader, config, start_epoch=0, noise_for_imgs=None, debug=False):
    epochs = config['train']['process']['epochs']
    device = config['device']
    save_model_every_n_epochs, save_img_every_n_epochs, save_img_path, save_dir = prepare_saving(model, config, start_epoch, debug)
    
    if noise_for_imgs is None:
        noise_for_imgs = torch.randn(size=(config['checkpoint']['img_B'], model.in_channels, model.img_size, model.img_size)).to(device)
    else:
        noise_for_imgs = noise_for_imgs.to(device)

    model = model.train()
    avg_loss = None

    for epoch in range(start_epoch, epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            ema_model.update(model)
            total_num += B
            total_loss += batch_loss.item() * B

        avg_loss = total_loss / total_num
        save_checkpoint(epoch + 1, epochs, save_model_every_n_epochs, model, ema_model, optimizer, scheduler, noise_for_imgs, avg_loss, save_dir)

        if (epoch + 1) == epochs or (save_img_every_n_epochs and (epoch + 1) % save_img_every_n_epochs == 0):
            save_img(model=ema_model.ema_model,
                     B=config['checkpoint']['img_B'],
                     T=config['checkpoint']['img_T'], 
                     path=save_img_path,
                     epoch=epoch + 1,
                     device=device,
                     noise_for_img=noise_for_imgs,
                     with_process=True)

        print(f"epoch {epoch + 1}/{epochs}. Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train_c':
        if args.experiment is None:
            exp_path = choose_experiment()
        else:
            exp_path = args.experiment
        model, ema_model, data_loader, optimizer, scheduler, epoch, best_loss, noise_for_imgs, config = load_train_checkpoint(exp_path, args)
    else:
        config = load_config(args)
        model = init_model(config)
        ema_model = init_ema(model, config)
        optimizer = init_optimizer(model, config)
        scheduler = init_scheduler(optimizer, config)
        data_loader = init_data_loader(config)
        epoch = 0
        noise_for_imgs = None
        avg_loss = None

    loss_instance = torch.nn.MSELoss()
    debug_mode = args.mode == 'debug'

    train_rectified_flow_model(model=model, ema_model=ema_model,
                               optimizer=optimizer, scheduler=scheduler,
                               criterion=loss_instance,
                               data_loader=data_loader,
                               config=config,
                               start_epoch=epoch,
                               noise_for_imgs=noise_for_imgs,
                               debug=debug_mode)
