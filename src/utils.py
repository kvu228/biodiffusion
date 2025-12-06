import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import logging
from datetime import datetime


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

    
def save_images_1D_to_2D(signals, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    imgs = signals.reshape(signals.shape[0], 1, 28, 28)
    print(imgs.shape)
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].imshow(imgs[i*5+j][k], cmap='gray')
    plt.savefig(path, format="jpeg")
    

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    if args.dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root=args.dataset_path,
           train=True, 
           transform=torchvision.transforms.ToTensor(),
           download=True)
    elif args.dataset_name == "Cifar10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("checkpoint", run_name), exist_ok=True)
    os.makedirs(os.path.join("logs", run_name), exist_ok=True)

    # Tạo log file với timestamp
    log_filename = os.path.join("logs", run_name, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    

    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Setup logging với format chi tiết
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # File handler - ghi vào file
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            # Console handler - hiển thị trên console
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Logging initialized. Log file: {log_filename}")
    logging.info(f"Run name: {run_name}")

    return log_filename
    
    
def save_signals(signals, path, **kwargs):
    """
    Save signals visualization. Supports both 1-channel (MITBIH) and 3-channel (UNIMIB) signals.
    
    Args:
        signals: Tensor of shape (n_samples, channels, seq_length)
        path: Path to save the image
    """
    signals = signals.to('cpu').detach().numpy()
    n_samples = min(signals.shape[0], 10)  # Show max 10 samples
    n_channels = signals.shape[1]
    
    if n_channels == 1:
        # Single channel (MITBIH) - original behavior
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    axs[i, j].plot(signals[idx][0][:])
                    axs[i, j].set_title(f'Sample {idx}')
                axs[i, j].grid(True, alpha=0.3)
    elif n_channels == 3:
        # Three channels (UNIMIB) - show ax, ay, az separately
        fig, axs = plt.subplots(2, 5, figsize=(20, 6))
        channel_names = ['ax', 'ay', 'az']
        colors = ['r', 'g', 'b']
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    for ch in range(3):
                        axs[i, j].plot(signals[idx][ch][:], 
                                      color=colors[ch], 
                                      label=channel_names[ch], 
                                      alpha=0.7)
                    axs[i, j].set_title(f'Sample {idx}')
                    axs[i, j].legend(loc='upper right', fontsize=6)
                axs[i, j].grid(True, alpha=0.3)
    else:
        # Generic multi-channel support
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    for ch in range(n_channels):
                        axs[i, j].plot(signals[idx][ch][:], alpha=0.7, label=f'Ch{ch}')
                    axs[i, j].set_title(f'Sample {idx}')
                    if n_channels <= 5:
                        axs[i, j].legend(loc='upper right', fontsize=6)
                axs[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, format="jpeg", dpi=150)
    plt.close()


def save_signals_cls_free(signals, labels, path, **kwargs):
    """
    Save signals with class labels visualization. 
    Supports both 1-channel (MITBIH) and 3-channel (UNIMIB) signals.
    
    Args:
        signals: Tensor of shape (n_samples, channels, seq_length)
        labels: Tensor or array of class labels
        path: Path to save the image
    """
    signals = signals.to('cpu').detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.to('cpu').numpy()
    
    n_samples = min(signals.shape[0], 10)  # Show max 10 samples
    n_channels = signals.shape[1]
    
    if n_channels == 1:
        # Single channel (MITBIH) - original behavior
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    axs[i, j].plot(signals[idx][0][:])
                    axs[i, j].set_title(f'Class {labels[idx]}')
                axs[i, j].grid(True, alpha=0.3)
    elif n_channels == 3:
        # Three channels (UNIMIB) - show ax, ay, az separately
        fig, axs = plt.subplots(2, 5, figsize=(20, 6))
        channel_names = ['ax', 'ay', 'az']
        colors = ['r', 'g', 'b']
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    for ch in range(3):
                        axs[i, j].plot(signals[idx][ch][:], 
                                      color=colors[ch], 
                                      label=channel_names[ch], 
                                      alpha=0.7)
                    axs[i, j].set_title(f'Class {labels[idx]}')
                    axs[i, j].legend(loc='upper right', fontsize=6)
                axs[i, j].grid(True, alpha=0.3)
    else:
        # Generic multi-channel support
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_samples:
                    for ch in range(n_channels):
                        axs[i, j].plot(signals[idx][ch][:], alpha=0.7, label=f'Ch{ch}')
                    axs[i, j].set_title(f'Class {labels[idx]}')
                    if n_channels <= 5:
                        axs[i, j].legend(loc='upper right', fontsize=6)
                axs[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, format="jpeg", dpi=150)
    plt.close()
    

    
    
def save_signals_cond_cls_free(sampled_signals, org_signals, cond_signals, labels, path, **kwargs):
    """
    Save conditional signals visualization (original, conditional, sampled).
    Supports both 1-channel (MITBIH) and 3-channel (UNIMIB) signals.
    
    Args:
        sampled_signals: Generated signals, shape (n_samples, channels, seq_length)
        org_signals: Original signals, shape (n_samples, channels, seq_length)
        cond_signals: Conditional signals, shape (n_samples, channels, seq_length)
        labels: Class labels
        path: Path to save the image
    """
    sampled_signals = sampled_signals.to('cpu').detach().numpy()
    org_signals = org_signals.to('cpu').detach().numpy()
    cond_signals = cond_signals.to('cpu').detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.to('cpu').numpy()
    
    n_samples = min(sampled_signals.shape[0], 5)  # Show max 5 samples
    n_channels = sampled_signals.shape[1]
    
    if n_channels == 1:
        # Single channel (MITBIH) - original behavior
        fig, axs = plt.subplots(3, 5, figsize=(20, 9))
        for i in range(5):
            if i < n_samples:
                axs[0, i].plot(org_signals[i][0][:], label='Original')
                axs[1, i].plot(cond_signals[i][0][:], label='Conditional')
                axs[2, i].plot(sampled_signals[i][0][:], label='Sampled')
                axs[0, i].set_title(f'Class {labels[i]}')
            for row in range(3):
                axs[row, i].grid(True, alpha=0.3)
                if i < n_samples:
                    axs[row, i].legend(loc='upper right', fontsize=6)
    elif n_channels == 3:
        # Three channels (UNIMIB) - show ax, ay, az separately
        fig, axs = plt.subplots(3, 5, figsize=(20, 9))
        channel_names = ['ax', 'ay', 'az']
        colors = ['r', 'g', 'b']
        row_labels = ['Original', 'Conditional', 'Sampled']
        
        for i in range(5):
            if i < n_samples:
                for row in range(3):
                    for ch in range(3):
                        if row == 0:
                            data = org_signals[i][ch][:]
                        elif row == 1:
                            data = cond_signals[i][ch][:]
                        else:
                            data = sampled_signals[i][ch][:]
                        
                        axs[row, i].plot(data, 
                                        color=colors[ch], 
                                        label=channel_names[ch], 
                                        alpha=0.7)
                    axs[row, i].set_title(f'{row_labels[row]} - Class {labels[i]}' if row == 0 else row_labels[row])
                    axs[row, i].legend(loc='upper right', fontsize=6)
            for row in range(3):
                axs[row, i].grid(True, alpha=0.3)
    else:
        # Generic multi-channel support
        fig, axs = plt.subplots(3, 5, figsize=(20, 9))
        for i in range(5):
            if i < n_samples:
                for row in range(3):
                    for ch in range(n_channels):
                        if row == 0:
                            data = org_signals[i][ch][:]
                        elif row == 1:
                            data = cond_signals[i][ch][:]
                        else:
                            data = sampled_signals[i][ch][:]
                        axs[row, i].plot(data, alpha=0.7, label=f'Ch{ch}')
                    axs[row, i].set_title(f'Class {labels[i]}' if row == 0 else '')
                    if n_channels <= 5:
                        axs[row, i].legend(loc='upper right', fontsize=6)
            for row in range(3):
                axs[row, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, format="jpeg", dpi=150)
    plt.close()    
    
def save_images_1D_to_2D_cls_free(signals, labels, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    imgs = signals.reshape(signals.shape[0], 1, 28, 28)
    print(imgs.shape)
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].imshow(imgs[i*5+j][k], cmap='gray')
            axs[i, j].set_title(f'{labels[i*5+j]}')
    plt.savefig(path, format="jpeg")
            

def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pt"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pt"))