from time import time
import torch
from tqdm import tqdm

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

def train_rectified_flow_model(model, optimizer, epochs, criterion, data_loader, device="cpu"):
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
        
        print(f"epoch {epoch + 1}/{epochs}. Loss: {avg_loss:.4f}")
    return avg_loss

def sample(model, B, T, device):
        """
        input: T: int number, num of steps
               B: int number, num of samples
        """
        model = model.to(device)
        model.eval()

        sample = torch.randn(B, model.in_channels, model.img_size, model.img_size).to(device)
        t = torch.linspace(0, 1, T).to(device)
        for i in range(len(t) - 1):
            t_curr = torch.ones(size=(B, 1)).to(device) * t[i]
            dt = t[i + 1] - t[i]

            sample = sample + dt * model(sample, t_curr)

        return sample 
  
