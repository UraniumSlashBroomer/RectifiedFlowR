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

def train_rectified_flow_model(model, optimizer, epochs, criterion, data_loader):
    model = model.train()
    avg_loss = None
    
    for epoch in range(epochs):
        total_loss, total_num = 0.0, 0.0
        for x in tqdm(data_loader):
            B, C, N, N = x.shape
            x = x.float()
            noise = torch.randn(size=(B, C, N, N))
            t = torch.rand(size=(B, 1, 1, 1))
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
