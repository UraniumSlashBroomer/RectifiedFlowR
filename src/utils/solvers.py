import torch

def euler_solver(model, T, device, sample, with_process=False):
    t = torch.linspace(0, 1, T).to(device) # [T]
    B, C, img_size, img_size = sample.shape
    output = torch.zeros_like(sample)

    if with_process:
        output = torch.zeros(size=(T, B, C, img_size, img_size))
        output[0] = sample


    for i in range(len(t) - 1):
        t_curr = torch.ones(size=(B, 1, 1)).to(device) * t[i] # [B, 1] * T = [B, T]
        dt = t[i + 1] - t[i]

        sample = sample + dt * model(sample, t_curr)
        if with_process:
            output[i + 1, :, :, :] = sample
    
    if with_process:
        return output
    return sample

