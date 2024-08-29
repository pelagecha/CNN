import torch

def select_processor():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device('cuda')
    else:
        print("Using CPU")
        return torch.device('cpu')
