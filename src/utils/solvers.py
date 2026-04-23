import torch
import torchdiffeq

def euler_solver(model, T, device, sample, with_process=False):
    """
    output shape if with_process:
        [T, B, C, H, W]
    else:
        [B, C, H, W] (only final result)
    """

    t = torch.linspace(0, 1, T).to(device) # [T]
    B, C, img_size, img_size = sample.shape
    output = torch.zeros_like(sample)

    if with_process:
        output = torch.zeros(size=(T, B, C, img_size, img_size))
        output[0] = sample

    for i in range(len(t) - 1):
        t_curr = torch.ones(size=(B, 1, 1)).to(device) * t[i] # [B, 1, 1]
        dt = t[i + 1] - t[i] # scalar

        sample = sample + dt * model(sample, t_curr)
        if with_process:
            output[i + 1, :, :, :] = sample
    
    if with_process:
        return output
    return sample


def heun_solver(model, T, device, sample, with_process=False):
    t = torch.linspace(0, 1, T).to(device)

    B, C, img_size, img_size = sample.shape
    output = torch.zeros_like(sample)

    if with_process:
        output = torch.zeros(size=(T, B, C, img_size, img_size))
        output[0] = sample

    for i in range(len(t) - 1):
        t_curr = torch.ones(size=(B, 1, 1)).to(device) * t[i]
        dt = t[i + 1] - t[i]

        speed_1 = model(sample, t_curr)
        x_next = sample + dt * speed_1
        t_next = t_curr + dt
        speed_2 = model(x_next, t_next)
        sample = sample + 0.5 * (speed_1 + speed_2) * dt

        if with_process:
            output[i + 1] = sample

    if with_process:
        return output
    return sample


def odeint_solver(model, device, sample):
    B = sample.shape[0]

    def odeint_func(t, x):
        t = torch.ones(B, 1, 1).to(device) * t
        return model(x, t)

    t_span = torch.tensor([0.0, 1.0], device=device)

    with torch.no_grad():
        trajectory = torchdiffeq.odeint(
                odeint_func,
                sample,
                t_span,
                rtol=1e-3,
                atol=1e-3,
                method='dopri5'
        )

    return trajectory[1]
