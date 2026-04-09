from rectified_flow import *
from data_utils import *
from utils import *
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    model = RectifiedFlowViT(img_size=32,
                             in_channels=3,
                             patch_size=4,
                             emb_dim=128,
                             ffn_dim_ratio=4,
                             n_heads=8,
                             num_layers=4).to(device)
    
    data_dict = get_CIFAR10_data()
    batch_size = 32
    epochs = 20
    optimizer = torch.optim.Adam(lr=3e-4, params=model.parameters(), weight_decay=1e-5)
    loss_instance = torch.nn.MSELoss()
    data_loader = torch.utils.data.DataLoader(data_dict["X_train"], batch_size=batch_size, shuffle=True, drop_last=True)

    train_rectified_flow_model(model=model, optimizer=optimizer, epochs=epochs, criterion=loss_instance, data_loader=data_loader, device=device)

    generated_img = sample(model=model, B=1, T=10, device=device).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    generated_img = (generated_img + 1) / 2
    generated_img = np.clip(generated_img, a_min=0, a_max=1)

    plt.imsave('images/generated_image.png', generated_img)

        
