from src.utils.initialization import load_eval_checkpoint
from src.utils.utils import *
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
            choices=['euler', 'heun', 'odeint'],
            default='odeint')

    parser.add_argument(
            '--mode', type=str,
            choices=['process', 'grid'],
            default='grid')

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
        if args.mode == 'grid':
            T = None
            try:
                rows, columns = map(int, input("rows, columns: ").split())
            except KeyboardInterrupt:
                break

            if args.solver != 'odeint':
                try:
                    T = int(input("T (steps): "))
                except KeyboardInterrupt:
                    break

            B = rows * columns
            imgs = sample(model=model,
                          B=B, T=T,
                          device=device,
                          solver=args.solver,
                          with_process=False) # [B, H, T, W, C], T = 1

            imgs = format_img(imgs)
            imgs = imgs.reshape(B, H, W, C)

            fig, axes = plt.subplots(rows, columns, figsize=(8, 8))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                img = imgs[i]
                axes[i].imshow(img)
                axes[i].axis('off')

            fig.tight_layout()
            fig.show()

        elif args.mode == 'process':
            assert args.solver != 'odeint', f"can't make process grid with odeint solver"
            try:
                B, T = map(int, input("B, T: ").split())
            except KeyboardInterrupt:
                break
            T_indexes = T_linspace(T)
            img = sample(model=model,
                         B=B, T=T,
                         device=device,
                         solver=args.solver,
                         T_indexes=T_indexes,
                         with_process=True) # [B, H, T, W, C]

            img = format_img(img)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)

            one_frame_w = img.shape[1] / len(T_indexes)
            tick_positions = np.arange(len(T_indexes)) * one_frame_w + one_frame_w / 2
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(f"T={i + 1}" for i in T_indexes)
            ax.set_yticks([])

            fig.tight_layout()
            fig.show()
