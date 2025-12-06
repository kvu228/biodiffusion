"""Component for displaying 1D signals"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt


def display_signals(signals, labels=None, n_cols=4, figsize=(15, 10)):
    """
    Display 1D signals in a grid
    
    Args:
        signals: Tensor of shape (batch, channels, seq_length)
        labels: Optional labels for each signal
        n_cols: Number of columns in the grid
        figsize: Figure size
    """
    signals = signals.cpu().detach().numpy() if isinstance(signals, torch.Tensor) else signals
    
    batch_size = signals.shape[0]
    channels = signals.shape[1]
    
    n_rows = (batch_size + n_cols - 1) // n_cols
    
    # Create subplots - always use 2D layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    for i in range(batch_size):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot all channels
        for ch in range(channels):
            ax.plot(signals[i, ch, :], label=f'Channel {ch}' if channels > 1 else None)
        
        if labels is not None:
            if isinstance(labels, (list, np.ndarray, torch.Tensor)):
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                label_val = labels[i] if i < len(labels) else None
            else:
                label_val = labels
            if label_val is not None:
                ax.set_title(f'Class {label_val}', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
    
    # Hide unused subplots
    for i in range(batch_size, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_signal_single(signal, label=None, figsize=(10, 4)):
    """
    Display a single 1D signal
    
    Args:
        signal: Tensor of shape (channels, seq_length) or (seq_length,)
        label: Optional label
        figsize: Figure size
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().detach().numpy()
    
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    elif signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
        signal = signal.T
    
    channels, seq_length = signal.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for ch in range(channels):
        ax.plot(signal[ch, :], label=f'Channel {ch}' if channels > 1 else None)
    
    if label is not None:
        ax.set_title(f'Class {label}', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    if channels > 1:
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

