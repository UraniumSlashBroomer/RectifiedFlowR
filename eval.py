from src.utils.initialization import load_checkpoint
from src.utils.utils import sample
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--device', type=str,
            default='cpu',
            help='which device to use on local machine')
    
    return parser.parse_args()


def find_experiments(root_dir='results'):
    root = pathlib.Path(root_dir)
    experiments = []

    for checkpoint_path in root.rglob("checkpoint.pth"):
        experiment_path = checkpoint_path.parent
        #if experiment_path.name != 'debug':
        experiments.append((experiment_path, experiment_path.parent))

    experiments.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return experiments


def load_experiment(experiment_path):
    config_path = experiment_path / 'resolved_config.yaml'
    checkpoint_path = experiment_path / 'checkpoint.pth'

    _, ema_model, _, _, _, _, config = load_checkpoint(config_path, checkpoint_path)
    model = ema_model.ema_model
    return model, config
    

if __name__ == '__main__':
    args = parse_args()
    device = args.device

    experiments = find_experiments('./results')
    
    print(f"choose which exp load:")
    for ind, experiment in enumerate(experiments):
        print(f"{ind + 1}. {experiment}")
    
    choosed_ind = int(input()) - 1
    exp_path = experiments[choosed_ind][0]
    model, config = load_experiment(exp_path)
    
    H = W = model.img_size
    C = model.in_channels

    while True:
        B, T = map(int, input().split())
        imgs = sample(model, B, T, device, with_process=True)
        imgs = imgs.reshape(B * H, T * W, C)
        imgs = np.clip(imgs, a_min=-1, a_max=1)
        imgs = 255 * ((imgs + 1) / 2)
        imgs = imgs.astype(np.uint8)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(imgs)
        
        one_frame_w = imgs.shape[1] / T 
        tick_positions = np.arange(T) * one_frame_w + one_frame_w / 2
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(f"T={i}" for i in range(T))
        ax.set_yticks([])

        fig.show()
