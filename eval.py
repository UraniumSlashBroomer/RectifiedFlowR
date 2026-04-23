from src.utils.initialization import load_eval_checkpoint
from src.utils.utils import sample, choose_experiment
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--device', type=str,
            default='cpu',
            help='which device to use on local machine')
    
    parser.add_argument(
            '--solver', type=str,
            default='odeint')

    return parser.parse_args()


def load_experiment(experiment_path, args):
    ema_model, config = load_eval_checkpoint(experiment_path, args)
    model = ema_model.ema_model
    return model, config
 

if __name__ == '__main__':
    args = parse_args()
    device = args.device
    
    exp_path = choose_experiment()
    model, config = load_experiment(exp_path, args)
    
    H = W = model.img_size
    C = model.in_channels

    while True:
        if args.solver == 'odeint':
            B = int(input("B: "))
            T = 0
        else:
            B, T = map(int, input("B, T: ").split())
        imgs = sample(model, B, T, device, solver=args.solver, with_process=False)
        imgs = imgs.reshape(B * H, W, C)

        # imgs = imgs.reshape(B * H, T * W, C)
        imgs = np.clip(imgs, a_min=-1, a_max=1)
        imgs = 255 * ((imgs + 1) / 2)
        imgs = imgs.astype(np.uint8)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(imgs)
        
        # one_frame_w = imgs.shape[1] / T 
        # tick_positions = np.arange(T) * one_frame_w + one_frame_w / 2
        # ax.set_xticks(tick_positions)
        # ax.set_xticklabels(f"T={i}" for i in range(T))
        # ax.set_yticks([])

        fig.show()
