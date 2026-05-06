import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from .solvers import *


def format_img(img):
    """
    shape [B, H, T, W, C] -> [B * H, T * W, C]
    pixel value [-1, -1] -> [0, 255]
    """

    B, H, T, W, C = img.shape
    img = img.reshape(B * H, T * W, C)
    img = np.clip(img, a_min=-1, a_max=1)
    img = 255 * ((img + 1) / 2)
    img = img.astype(np.uint8)

    return img


def T_linspace(T):
    """
    function for plots where will be process of generation
    """
    if T > 10:
        return np.linspace(0, T - 1, 10).round().astype('int')
    else:
        return np.linspace(0, T - 1, T).astype('int')


@torch.no_grad()
def sample_and_save(model, B, device, path, title, T=None, file_name=None, noise_for_img=None, solver='euler', with_process=False):
    T_indexes = T_linspace(T) if with_process else None
    img = sample(model=model,
                  B=B, T=T,
                  device=device,
                  noise=noise_for_img,
                  solver=solver,
                  T_indexes=T_indexes,
                  with_process=with_process)
    
    img = format_img(img)
    
    save_img(img, path, title, file_name, T_indexes, with_process)


def save_img(img, path, title, file_name=None, T_indexes=None, with_process=False):
    """
    input:
    - img [B, H, T, W, C]
    - path (where to save the img)
    - title (save name and title for plt if flag save_with_plt)
    - save_with_plt (flag for saving plot)
    """
    result_path = os.path.join(path, f"{file_name if file_name is not None else title}.png")
    
    if with_process and T_indexes is not None:
        save_process(img, T_indexes, result_path, title)
    else:
        img = Image.fromarray(img)
        # img = img.resize((img.width * 4, img.height * 4), resample=Image.Resampling.NEAREST)
        img.save(result_path)
            

def save_process(grid, T_indexes, result_path, title):
    _, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(grid)
    
    one_frame_w = grid.shape[1] / len(T_indexes)
    tick_positions = np.arange(len(T_indexes)) * one_frame_w + one_frame_w / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(f"T={i + 1}" for i in T_indexes)
    ax.set_yticks([])

    plt.title(f"{title}")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def sample(model, B, device, T=None, noise=None, solver='euler', T_indexes=None, with_process=False):
    """
    input:
    - model
    - B (batch_size)
    - T (for Euler and Heun solver, number of steps)
    - solver (solver type)
    - with_process (for Euler and Heun solver, for better visualization)

    output:
    - output (numpy array on cpu) [B, H, T, W, C]
    """

    model = model.to(device)
    model.eval()
    
    if noise is None:
        sample = torch.randn(B, model.in_channels, model.img_size, model.img_size).to(device)
    else:
        sample = noise

    if with_process and T_indexes is None:
        T_indexes = T_linspace(T)
    
    if solver == 'euler':
        output = euler_solver(model, T, device, sample, T_indexes, with_process) 
    elif solver == 'heun':
        output = heun_solver(model, T, device, sample, T_indexes, with_process)
    elif solver == 'odeint':
        with_process = False
        output = odeint_solver(model, device, sample)
    else:
        print(f"please, provide possible solver")
        raise ValueError 

    if not(with_process):
        output = output.unsqueeze(0) # [T, B, C, H, W], T = 1
    return output.permute(1, 3, 0, 4, 2).cpu().detach().numpy() # [T, B, C, H, W] -> [B, H, T, W, C]


def find_experiments(root_dir='results'):
    root = pathlib.Path(root_dir)
    experiments = []

    for checkpoint_path in root.rglob("checkpoint.pth"):
        experiment_path = checkpoint_path.parent
        experiments.append((experiment_path, experiment_path.parent))

    experiments.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return experiments


def get_config_checkpoint_path(experiment_path):
    if type(experiment_path) == str:
        experiment_path = pathlib.Path(experiment_path)

    config_path = experiment_path / 'resolved_config.yaml'
    checkpoint_path = experiment_path / 'checkpoint.pth'
    return config_path, checkpoint_path


def choose_experiment():
    experiments = find_experiments('./results')
    
    print(f"choose experiment to load:")
    for ind, experiment in enumerate(experiments):
        print(f"{ind + 1}. {experiment[1].name} / {experiment[0].name}")
    
    choosed_ind = int(input()) - 1
    exp_path = experiments[choosed_ind][0]
    return exp_path
