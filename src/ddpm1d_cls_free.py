import os
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from utils import setup_logging, save_signals_cls_free, save_checkpoint
from modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
import logging
from torch.utils.tensorboard import SummaryWriter
from MITBIH import mitbih_allClass
from torch.utils import data
from time import perf_counter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
data_path = "datasets/heartbeat/mitbih_train.csv"

class TrainingDataset(data.Dataset):
    """
    Custom PyTorch Dataset for training.
    """
    def __init__(self, filename):
        self.fiveClassECG = mitbih_allClass(filename=filename)

    def __len__(self):
        return len(self.fiveClassECG)

    def __getitem__(self, idx):
        signals, labels = self.fiveClassECG[idx]
        return signals, labels

def train(run_name: str = 'DDPM1D_cls_free_MITBIH',
          epochs: int = 300,
          batch_size: int = 64,
          seq_length: int = 128,
          num_classes: int = 5,
          lr: float = 1e-4,
          device: str = 'cuda',
          num_workers: Optional[int] = None,
          resume_from_checkpoint: bool = True,
          early_stopping: bool = True,
          patience: int = 20,
          min_delta: float = 1e-6,
          monitor: str = 'loss'):
    """
    Training function for the Deep Diffusion Probabilistic Model (DDPM) on 1D signals with classification.
    """
    import platform
    # On Windows, num_workers > 0 can cause DataLoader to hang
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 1
    
    setup_logging(run_name)
    device = device
    logging.info(f"Using device: {device}")
    logging.info(f"Using num_workers: {num_workers}")
    dataset = TrainingDataset(filename=data_path)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    classes = num_classes

    # Define the UNet model
    model = Unet1D_cls_free(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        cond_drop_prob=0.5,
        channels=1
    ).to(device)

    # Define the Gaussian Diffusion model
    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length=seq_length,
        timesteps=1000
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    logger = SummaryWriter(os.path.join("runs", run_name))
    dataloader_len = len(dataloader)

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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        start_time = perf_counter()
        logging.info(f"Starting epoch {epoch + 1}/{epochs}:")
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", mininterval=0.1, maxinterval=1.0)

        for i, (signals, labels) in enumerate(pbar):
            if isinstance(signals, np.ndarray):
                signals = torch.from_numpy(signals).float()
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()

            signals = signals.to(device).float()
            labels = labels.to(device).long()

            # Calculate loss using the diffusion model
            loss = diffusion(signals, classes=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{np.mean(epoch_losses):.4f}',
            }, refresh=True)

            logger.add_scalar("loss", loss.item(), global_step=epoch * dataloader_len + i)

        avg_epoch_loss = np.mean(epoch_losses)
        end_time = perf_counter()
        logging.info(f"Epoch {epoch + 1}/{epochs} completed | Average Loss: {avg_epoch_loss:.4f} | Time elapsed: {(end_time - start_time):.2f}s")
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

            # Kiểm tra early stop
            if patience_counter >= patience:
                logging.info(f"\n{'='*60}")
                logging.info("Early stopping triggered!")
                logging.info(f"Best loss: {best_loss:.6f} at epoch {best_epoch}")
                logging.info(f"No improvement for {patience} epochs")
                logging.info(f"{'='*60}\n")
                break

        # Generate and save sampled signals
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).to(device)
        sampled_signals = diffusion.sample(
            classes=labels,
            cond_scale=3.
        )

        # Save the generated signals as images
        save_signals_cls_free(sampled_signals, labels, os.path.join("results", run_name, f"{epoch + 1}.jpg"))

        # Save model checkpoint
        is_best = False
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'avg_loss': avg_epoch_loss,
            'best_loss': best_loss if early_stopping else None,
            'best_epoch': best_epoch if early_stopping else None,
            'patience_counter': patience_counter if early_stopping else None,
            'optimizer': optimizer.state_dict(),
        }, is_best, output_dir=os.path.join("checkpoint", run_name))
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
    Launch the training process with predefined parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        epochs=300,
        device=device,
        num_workers=0,  # Set to 0 on Windows to avoid DataLoader hanging
    )

if __name__ == '__main__':
    launch()
