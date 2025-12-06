import os
import sys
sys.path.insert(0, './modules/')
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils_cond import save_signals, save_checkpoint, setup_logging
from modules.modules1D_cond import Unet1D, GaussianDiffusion1D
import logging
from torch.utils.tensorboard import SummaryWriter
from UNIMIB import unimib_masked
from torch.utils import data
import random
from typing import Optional
from time import perf_counter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
data_path = "datasets/unimib/unimib_train.csv"

class TrainingDataset(data.Dataset):
    """
    Custom PyTorch Dataset for UNIMIB training with conditional data.
    """
    def __init__(self, filename, class_id):
        self.cond_unimib = unimib_masked(filename=filename, class_id=class_id)

    def __len__(self):
        return len(self.cond_unimib)

    def __getitem__(self, idx):
        data_dict = {
            'org_data': self.cond_unimib[idx]['org_data'],
            'cond_data': self.cond_unimib[idx]['cond_data']
        }
        return data_dict

def train(run_name: str = 'DDPM1D_SelfConditional_UNIMIB',
          epochs: int = 100,
          batch_size: int = 32,
          seq_length: int = 128,
          lr: float = 1e-4,
          sample_size: int = 10,
          device: str = 'cuda',
          num_workers: Optional[int] = None,
          resume_from_checkpoint: bool = True,
          early_stopping: bool = True,
          patience: int = 10,
          min_delta: float = 1e-6,
          monitor: str = 'loss',
          class_id: int = 0):
    """
    Training function for UNIMIB dataset (3-channel accelerometer signals) with signal conditioning.
    
    According to paper:
    - Channels: 3 (ax, ay, az)
    - Batch size: 32
    - Learning rate: 1e-4 (for signal conditional)
    - Epochs: 100
    - Optimizer: Adam
    - Objective: pred_v
    - Timesteps: 2000 (paper) / 1000 (code default)
    """
    import platform
    if num_workers is None:
        num_workers = 0 if platform.system() == "Windows" else 1

    setup_logging(run_name)
    device = device
    logging.info(f"Using device: {device}")
    logging.info(f"Using num_workers: {num_workers}")
    logging.info(f"Training UNIMIB signal conditional model for class {class_id}")
    
    dataset = TrainingDataset(filename=data_path, class_id=class_id)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Define the UNet model - UNIMIB has 3 channels
    model = Unet1D(
        dim=64,
        self_condition=True,
        dim_mults=(1, 2, 4, 8),  # According to paper
        channels=3                # 3 channels for accelerometer (ax, ay, az)
    ).to(device)

    # Define the Gaussian Diffusion model
    # Note: Paper uses 2000 timesteps, but code default is 1000
    # You can change to 2000 if needed: timesteps=2000
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_length,
        timesteps=1000,  # Can change to 2000 per paper
        objective='pred_v'  # v-parameterization according to paper
    ).to(device)

    # Use Adam optimizer according to paper
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = SummaryWriter(os.path.join("runs", run_name))
    dataloader_len = len(dataloader)
    sample_size = sample_size

    # ========== EARLY STOPPING SETUP ==========
    if early_stopping:
        best_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        logging.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}, monitor={monitor}")

    # ========== RESUME FROM CHECKPOINT ==========
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoint", run_name, "checkpoint.pt")
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict())

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 0

        # ========== Restore early stopping state ==========
        if early_stopping:
            if 'best_loss' in checkpoint and checkpoint['best_loss'] is not None:
                best_loss = checkpoint['best_loss']
            if 'best_epoch' in checkpoint and checkpoint['best_epoch'] is not None:
                best_epoch = checkpoint['best_epoch']
            if 'patience_counter' in checkpoint and checkpoint['patience_counter'] is not None:
                patience_counter = checkpoint['patience_counter']
            logging.info(f"Restored early stopping: best_loss={best_loss:.6f}, best_epoch={best_epoch}, patience={patience_counter}")
        logging.info(f"Checkpoint loaded: Resuming from epoch {start_epoch}")

    # ========== TRAINING LOOP ==========
    for epoch in range(start_epoch, epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}:")
        start_time_epoch = perf_counter()
        epoch_losses = []
        pbar = tqdm(dataloader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    mininterval=0.1,
                    maxinterval=1.0)

        for i, data_dict in enumerate(pbar):
            sig1 = data_dict['org_data'].to(device).to(torch.float)
            sig2 = data_dict['cond_data'].to(device).to(torch.float)

            # Calculate loss using the diffusion model
            loss = diffusion(sig1, sig2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg_loss": f"{np.mean(epoch_losses):.4f}"
            }, refresh=True)

            logger.add_scalar("loss", loss.item(), global_step=epoch * dataloader_len + i)

        end_time_epoch = perf_counter()
        avg_epoch_loss = np.mean(epoch_losses)
        logging.info(f"Epoch {epoch + 1}/{epochs} completed | Average Loss: {avg_epoch_loss:.4f} | Time elapsed: {(end_time_epoch - start_time_epoch):.2f}s")
        logger.add_scalar("epoch_loss", avg_epoch_loss, epoch)

        # ========== EARLY STOPPING CHECK ==========
        if early_stopping:
            if avg_epoch_loss < (best_loss - min_delta):
                best_loss = avg_epoch_loss
                best_epoch = epoch + 1
                patience_counter = 0

                # Lưu best model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'avg_loss': avg_epoch_loss,
                    'best_loss': best_loss if early_stopping else None,
                    'best_epoch': best_epoch if early_stopping else None,
                    'patience_counter': patience_counter if early_stopping else None,
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, output_dir=os.path.join("checkpoint", run_name))

                logging.info(f"✓ Model improved! New best loss: {best_loss:.6f} (epoch {best_epoch})")
            else:
                patience_counter += 1
                logging.info(f"  No improvement. Patience: {patience_counter}/{patience} (best: {best_loss:.6f} at epoch {best_epoch})")

        # Generate and save sampled signals
        index_list = [i for i in range(len(dataset))]
        random.shuffle(index_list)
        # Get conditional data from dataset
        cond_data_list = []
        for idx in index_list[:sample_size]:
            cond_data_list.append(dataset.cond_unimib[idx]['cond_data'])
        cond_data = torch.stack(cond_data_list).to(device).to(torch.float)
        sampled_signals = diffusion.sample(batch_size=sample_size, input_cond=cond_data)
        # Shape: (sample_size, 3, 128) for UNIMIB

        is_best = False

        save_signals(sampled_signals, os.path.join("results", run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'avg_loss': avg_epoch_loss,
            'best_loss': best_loss if early_stopping else None,
            'best_epoch': best_epoch if early_stopping else None,
            'patience_counter': patience_counter if early_stopping else None,
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", run_name))
        logging.info(f"Checkpoint saved at epoch {epoch + 1}")

    # ========== FINAL SUMMARY ==========
    if early_stopping:
        final_epoch = epoch + 1 if 'epoch' in locals() else start_epoch
        logging.info(f"\n{'='*60}")
        logging.info("Training completed!")
        logging.info(f"Best model: epoch {best_epoch} with loss {best_loss:.6f}")
        logging.info(f"Total epochs trained: {final_epoch}")
        logging.info(f"{'='*60}\n")


def launch():
    """
    Launch the training process with predefined parameters for UNIMIB.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        epochs=100,
        device=device,
        class_id=0  # Train for class 0, can change to other classes
    )


if __name__ == '__main__':
    launch()

