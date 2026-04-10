from time import time
import torch
from tqdm import tqdm
from datetime import datetime
import os
import yaml
from PIL import Image
import numpy as np


def time_inference(model, batch_size, n_batches):
    A = torch.randn(size=(batch_size, 3, model.img_size, model.img_size))

    start = time()
    _ = model(A)
    end = time()
    diff = end - start
    print(f"time for 1 batch with size {batch_size}: {diff // 60} min {diff % 60} sec")

    start = time()
    for _ in range(n_batches):
        output = model(A)
    end = time()
    
    diff = end - start
    print(f"tiem for {n_batches} batches with size {batch_size}: {diff // 60} min {diff % 60} sec")

def train_rectified_flow_model(model, optimizer, criterion, data_loader, config):
    
    epochs = config['train']['process']['epochs']
    device = config['device']
    save_model_every_n_epochs = config['checkpoint']['model_save_every_n_epochs']

    save_img_every_n_epochs = config['checkpoint']['img_save_every_n_epochs']
    noise_for_imgs = None
    if save_img_every_n_epochs:
        noise_for_imgs = torch.randn(size=(config['checkpoint']['img_B'], model.in_channels, model.img_size, model.img_size))

    # savedir and save config
    run_id_part_1 = datetime.now().strftime("%Y-%m-%d")
    run_id_part_2 = datetime.now().strftime("%H-%M-%S") + f"ViT-{model.n_heads}_{model.patch_size}_lr_{config['train']['optimizer']['lr']}_ep_{epochs}"
    run_id = os.path.join(run_id_part_1, run_id_part_2)
    save_dir = os.path.join("results", run_id)
    os.makedirs(save_dir, exist_ok=True)
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
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "noise_for_imgs": noise_for_imgs,
                      "best_score": avg_loss,
        }

        if save_model_every_n_epochs and ((epoch + 1) == epochs or (epoch + 1) % save_model_every_n_epochs == 0):
            torch.save(checkpoint, os.path.join(save_dir, "checkpoint_latest.pth"))

        if save_img_every_n_epochs and ((epoch + 1) == epochs or (epoch + 1) % save_img_every_n_epochs == 0):
            save_img(model=model,
                     B=config['checkpoint']['img_B'],
                     T=config['checkpoint']['img_T'], 
                     path=save_img_path,
                     epoch=epoch + 1,
                     device=device)

        print(f"epoch {epoch + 1}/{epochs}. Loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def save_img(model, B, T, device, path, epoch, with_process=True):
    result_path = os.path.join(path, f"{epoch}.png")

    imgs = sample(model, B, T, device, with_process)
    H, B, T, W, C = imgs.shape

    imgs = imgs.reshape(H * B, T * W, C)
    imgs = np.clip(imgs, a_min=-1, a_max=1)
    imgs = 255 * ((imgs + 1) / 2)
    imgs = imgs.astype(np.uint8)

    imgs = Image.fromarray(imgs)
    imgs = imgs.resize((imgs.width * 4, imgs.height * 4), resample=Image.Resampling.NEAREST)
    imgs.save(result_path)

@torch.no_grad()
def sample(model, B, T, device, with_process=False):
        """
        input: T: int number, num of steps
               B: int number, num of samples
        """
        model = model.to(device)
        model.eval()

        sample = torch.randn(B, model.in_channels, model.img_size, model.img_size).to(device)
        t = torch.linspace(0, 1, T).to(device) # [T]

        if with_process:
            output = torch.zeros(size=(T, B, model.in_channels, model.img_size, model.img_size)).to(device)
            output[0, :, :, :] = sample

        for i in range(len(t) - 1):
            t_curr = torch.ones(size=(B, 1, 1)).to(device) * t[i] # [B, 1] * T = [B, T]
            dt = t[i + 1] - t[i]

            sample = sample + dt * model(sample, t_curr)
            if with_process:
                output[i + 1, :, :, :] = sample
        
        if not(with_process):
            return sample.permute(0, 2, 3, 1).cpu().detach().numpy()
        return output.permute(3, 1, 0, 4, 2).cpu().detach().numpy() # [T, B, C, H, W] -> [H, B, T, W, C]
  
