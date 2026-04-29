import torch


def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: no GPU available, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: only {n_gpu} GPU(s) available, requested {n_gpu_use}.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    return device, list(range(n_gpu_use))
