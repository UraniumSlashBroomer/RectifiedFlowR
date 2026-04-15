import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def save_img(model, B, T, device, path, epoch, noise_for_img, with_process=True, save_with_plt=True):
    result_path = os.path.join(path, f"{epoch}.png")
    choosed_Ts = None

    imgs = sample(model=model,
                  B=B, T=T,
                  device=device,
                  noise=noise_for_img,
                  with_process=with_process)

    H, B, T, W, C = imgs.shape

    if T >= 11:
        choosed_Ts = np.linspace(0, T - 1, 10).round().astype('int')
        imgs = imgs[:, :, choosed_Ts, :, :]
        T = 10

    imgs = imgs.reshape(B * H, T * W, C)
    imgs = np.clip(imgs, a_min=-1, a_max=1)
    imgs = 255 * ((imgs + 1) / 2)
    imgs = imgs.astype(np.uint8)
    
    if save_with_plt:
        if choosed_Ts is None:
            choosed_Ts = range(0, T)
        save_img_with_plt(imgs, choosed_Ts, epoch, result_path)
        return

    imgs = Image.fromarray(imgs)
    imgs = imgs.resize((imgs.width * 4, imgs.height * 4), resample=Image.Resampling.NEAREST)
    imgs.save(result_path)


def save_img_with_plt(grid, choosed_Ts, epoch, result_path):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(grid)
    
    one_frame_w = grid.shape[1] / len(choosed_Ts)
    tick_positions = np.arange(len(choosed_Ts)) * one_frame_w + one_frame_w / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(f"T={i}" for i in choosed_Ts)
    ax.set_yticks([])

    plt.title(f"epoch {epoch}")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def sample(model, B, T, device, noise=None, with_process=False):
        """
        input: T: int number, num of steps
               B: int number, num of samples
        """
        model = model.to(device)
        model.eval()
        
        if noise is None:
            sample = torch.randn(B, model.in_channels, model.img_size, model.img_size).to(device)
        else:
            sample = noise

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
        return output.permute(1, 3, 0, 4, 2).cpu().detach().numpy() # [T, B, C, H, W] -> [B, H, T, W, C]
