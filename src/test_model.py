"""Test trained model with test dataset and calculate metrics from paper"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils import data
from pathlib import Path
import sys
import logging

try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False
    logging.warning("PyWavelets not installed. Wavelet Coherence will use alternative method.")

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from MITBIH import mitbih_allClass, reverse_cls_dit
from modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
from utils import save_signals_cls_free

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class TestDataset(data.Dataset):
    """Test dataset loader"""
    def __init__(self, filename):
        self.fiveClassECG = mitbih_allClass(filename=filename, isBalanced=False)

    def __len__(self):
        return len(self.fiveClassECG)

    def __getitem__(self, idx):
        signals, labels = self.fiveClassECG[idx]
        return signals, labels


def load_model(checkpoint_path, num_classes=5, seq_length=128, device='cuda', 
               dim=64, dim_mults=(1, 2, 4, 8), cond_drop_prob=0.5, channels=1):
    """Load trained model from checkpoint"""
    model = Unet1D_cls_free(
        dim=dim,
        dim_mults=dim_mults,
        num_classes=num_classes,
        cond_drop_prob=cond_drop_prob,
        channels=channels
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        if hasattr(checkpoint['model'], 'state_dict'):
            state_dict = checkpoint['model'].state_dict()
        else:
            state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length=seq_length,
        timesteps=1000
    ).to(device)
    diffusion.eval()
    
    return model, diffusion


def wavelet_coherence_score(real_signals, generated_signals, wavelet='db4', scales=None):
    """
    Calculate Wavelet Coherence Score between real and generated signals.
    
    Args:
        real_signals: numpy array of shape (n_samples, channels, seq_length)
        generated_signals: numpy array of shape (n_samples, channels, seq_length)
        wavelet: wavelet type for decomposition
        scales: scales for wavelet transform
    
    Returns:
        Average coherence score
    """
    if scales is None:
        scales = np.arange(1, 33)  # Default scales
    
    coherence_scores = []
    
    for i in range(len(real_signals)):
        real_sig = real_signals[i, 0, :] if real_signals.ndim == 3 else real_signals[i, :]
        gen_sig = generated_signals[i, 0, :] if generated_signals.ndim == 3 else generated_signals[i, :]
        
        # Normalize signals
        real_sig = (real_sig - real_sig.mean()) / (real_sig.std() + 1e-8)
        gen_sig = (gen_sig - gen_sig.mean()) / (gen_sig.std() + 1e-8)
        
        # Continuous Wavelet Transform
        try:
            if HAS_PYWAVELETS:
                coeffs_real, _ = pywt.cwt(real_sig, scales, wavelet)
                coeffs_gen, _ = pywt.cwt(gen_sig, scales, wavelet)
                
                # Calculate coherence (cross-spectrum normalized by power spectra)
                cross_spectrum = np.abs(coeffs_real * np.conj(coeffs_gen))
                power_real = np.abs(coeffs_real) ** 2
                power_gen = np.abs(coeffs_gen) ** 2
                
                coherence = cross_spectrum / (np.sqrt(power_real * power_gen) + 1e-8)
                coherence = np.clip(coherence, 0, 1)  # Ensure [0, 1]
                
                # Average coherence across all scales and time
                avg_coherence = np.mean(coherence)
            else:
                # Alternative: Use frequency domain correlation
                # FFT-based coherence
                fft_real = np.fft.fft(real_sig)
                fft_gen = np.fft.fft(gen_sig)
                
                cross_spectrum = np.abs(fft_real * np.conj(fft_gen))
                power_real = np.abs(fft_real) ** 2
                power_gen = np.abs(fft_gen) ** 2
                
                coherence = cross_spectrum / (np.sqrt(power_real * power_gen) + 1e-8)
                avg_coherence = np.mean(coherence)
            
            coherence_scores.append(avg_coherence)
        except Exception as e:
            logging.warning(f"Error computing coherence for sample {i}: {e}")
            coherence_scores.append(0.0)
    
    return np.mean(coherence_scores)


def discriminative_score(real_signals, generated_signals, classifier=None):
    """
    Calculate Discriminative Score using a classifier to distinguish real vs generated.
    
    Lower score = better (harder to distinguish)
    Score = 1 - accuracy of classifier
    
    Args:
        real_signals: numpy array of real signals
        generated_signals: numpy array of generated signals
        classifier: optional pre-trained classifier (if None, uses simple features)
    
    Returns:
        Discriminative score (0-1, lower is better)
    """
    n_real = len(real_signals)
    n_gen = len(generated_signals)
    
    # Combine real and generated signals
    all_signals = np.concatenate([real_signals, generated_signals], axis=0)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_gen)])
    
    # Flatten signals for feature extraction
    if all_signals.ndim == 3:
        all_signals_flat = all_signals.reshape(all_signals.shape[0], -1)
    else:
        all_signals_flat = all_signals
    
    # Simple feature-based classifier using statistical features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Extract statistical features
    features = []
    for sig in all_signals_flat:
        features.append([
            np.mean(sig),
            np.std(sig),
            np.var(sig),
            np.min(sig),
            np.max(sig),
            np.median(sig),
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            np.sum(np.abs(np.diff(sig))),  # Total variation
        ])
    
    features = np.array(features)
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Discriminative score = 1 - accuracy (lower is better)
    disc_score = 1.0 - accuracy
    
    return disc_score


def f1_score_metric(real_signals, generated_signals, real_labels, generated_labels):
    """
    Calculate F1-Score for classification task.
    Uses a classifier trained on real data to classify generated signals.
    
    Args:
        real_signals: numpy array of real signals with labels
        generated_signals: numpy array of generated signals with labels
        real_labels: labels for real signals
        generated_labels: labels for generated signals
    
    Returns:
        F1-score (macro and weighted average)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score as sklearn_f1
    
    # Prepare data
    if real_signals.ndim == 3:
        real_flat = real_signals.reshape(real_signals.shape[0], -1)
        gen_flat = generated_signals.reshape(generated_signals.shape[0], -1)
    else:
        real_flat = real_signals
        gen_flat = generated_signals
    
    # Train classifier on real data
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(real_flat, real_labels)
    
    # Predict on generated signals
    gen_pred = clf.predict(gen_flat)
    
    # Calculate F1 scores
    f1_macro = sklearn_f1(generated_labels, gen_pred, average='macro')
    f1_weighted = sklearn_f1(generated_labels, gen_pred, average='weighted')
    f1_per_class = sklearn_f1(generated_labels, gen_pred, average=None)
    
    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'predictions': gen_pred
    }


def test_model(checkpoint_path, test_data_path, num_classes=5, seq_length=128, 
               device='cuda', n_samples_per_class=100, cfg_scale=3.0,
               dim=64, dim_mults=(1, 2, 4, 8), cond_drop_prob=0.5, channels=1):
    """
    Test trained model and calculate metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test dataset CSV file
        num_classes: Number of classes
        seq_length: Sequence length
        device: Device to use
        n_samples_per_class: Number of samples to generate per class
        cfg_scale: Classifier-free guidance scale
        dim: Model dimension
        dim_mults: Dimension multipliers
        cond_drop_prob: Conditional dropout probability
        channels: Number of channels
    
    Returns:
        Dictionary with all metrics
    """
    logging.info("="*60)
    logging.info("Starting Model Testing")
    logging.info("="*60)
    
    # Load model
    logging.info(f"Loading model from: {checkpoint_path}")
    model, diffusion = load_model(
        checkpoint_path, num_classes, seq_length, device,
        dim, dim_mults, cond_drop_prob, channels
    )
    logging.info("Model loaded successfully!")
    
    # Load test dataset
    logging.info(f"Loading test dataset from: {test_data_path}")
    test_dataset = TestDataset(filename=test_data_path)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    logging.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Collect real signals and labels from test set
    real_signals_list = []
    real_labels_list = []
    
    logging.info("Collecting real signals from test set...")
    for signals, labels in tqdm(test_dataloader, desc="Loading test data"):
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        
        signals = signals.to(device)
        real_signals_list.append(signals.cpu().numpy())
        real_labels_list.append(labels.cpu().numpy())
    
    real_signals = np.concatenate(real_signals_list, axis=0)
    real_labels = np.concatenate(real_labels_list, axis=0).astype(np.int64)
    
    logging.info(f"Real signals shape: {real_signals.shape}")
    logging.info(f"Real labels distribution: {np.bincount(real_labels)}")
    
    # Generate signals for each class
    logging.info(f"Generating {n_samples_per_class} samples per class...")
    generated_signals_list = []
    generated_labels_list = []
    
    for class_id in range(num_classes):
        labels = torch.tensor([class_id] * n_samples_per_class, device=device)
        
        with torch.no_grad():
            gen_signals = diffusion.sample(classes=labels, cond_scale=cfg_scale)
        
        generated_signals_list.append(gen_signals.cpu().numpy())
        generated_labels_list.append([class_id] * n_samples_per_class)
        logging.info(f"Generated {n_samples_per_class} samples for class {class_id} ({reverse_cls_dit[class_id]})")
    
    generated_signals = np.concatenate(generated_signals_list, axis=0)
    generated_labels = np.array(generated_labels_list).flatten().astype(np.int64)
    
    logging.info(f"Generated signals shape: {generated_signals.shape}")
    
    # Match number of real and generated samples for fair comparison
    n_samples = min(len(real_signals), len(generated_signals))
    real_signals = real_signals[:n_samples]
    real_labels = real_labels[:n_samples]
    generated_signals = generated_signals[:n_samples]
    generated_labels = generated_labels[:n_samples]
    
    # Calculate metrics
    logging.info("="*60)
    logging.info("Calculating Metrics")
    logging.info("="*60)
    
    # 1. Wavelet Coherence Score
    logging.info("Calculating Wavelet Coherence Score...")
    coherence_score = wavelet_coherence_score(real_signals, generated_signals)
    logging.info(f"Wavelet Coherence Score: {coherence_score:.4f}")
    
    # 2. Discriminative Score
    logging.info("Calculating Discriminative Score...")
    disc_score = discriminative_score(real_signals, generated_signals)
    logging.info(f"Discriminative Score: {disc_score:.4f} (lower is better)")
    
    # 3. F1-Score
    logging.info("Calculating F1-Score...")
    f1_results = f1_score_metric(real_signals, generated_signals, real_labels, generated_labels)
    logging.info(f"F1-Score (Macro): {f1_results['f1_macro']:.4f}")
    logging.info(f"F1-Score (Weighted): {f1_results['f1_weighted']:.4f}")
    logging.info("F1-Score per class:")
    for i, f1 in enumerate(f1_results['f1_per_class']):
        logging.info(f"  Class {i} ({reverse_cls_dit[i]}): {f1:.4f}")
    
    # Save results
    results = {
        'wavelet_coherence': coherence_score,
        'discriminative_score': disc_score,
        'f1_macro': f1_results['f1_macro'],
        'f1_weighted': f1_results['f1_weighted'],
        'f1_per_class': f1_results['f1_per_class'],
        'n_samples': n_samples,
        'n_samples_per_class': n_samples_per_class,
    }
    
    # Save generated samples
    output_dir = Path("results") / Path(checkpoint_path).parent.name / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save some generated samples for visualization
    sample_labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device=device)
    with torch.no_grad():
        sample_signals = diffusion.sample(classes=sample_labels, cond_scale=cfg_scale)
    
    save_path = output_dir / "test_generated_samples.jpg"
    save_signals_cls_free(sample_signals, sample_labels.cpu().numpy(), str(save_path))
    logging.info(f"Generated samples saved to: {save_path}")
    
    # Save results to file
    results_path = output_dir / "test_metrics.txt"
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Testing Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Dataset: {test_data_path}\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Samples per class: {n_samples_per_class}\n")
        f.write(f"CFG Scale: {cfg_scale}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Wavelet Coherence Score: {coherence_score:.6f}\n")
        f.write(f"  Discriminative Score: {disc_score:.6f} (lower is better)\n")
        f.write(f"  F1-Score (Macro): {f1_results['f1_macro']:.6f}\n")
        f.write(f"  F1-Score (Weighted): {f1_results['f1_weighted']:.6f}\n")
        f.write("\nF1-Score per class:\n")
        for i, f1 in enumerate(f1_results['f1_per_class']):
            f.write(f"  Class {i} ({reverse_cls_dit[i]}): {f1:.6f}\n")
    
    logging.info(f"Results saved to: {results_path}")
    logging.info("="*60)
    logging.info("Testing completed!")
    logging.info("="*60)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, 
                       default='datasets/heartbeat/mitbih_test.csv',
                       help='Path to test dataset CSV file')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--seq_length', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to generate per class')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--dim', type=int, default=64,
                       help='Model dimension')
    
    args = parser.parse_args()
    
    results = test_model(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        num_classes=args.num_classes,
        seq_length=args.seq_length,
        device=args.device,
        n_samples_per_class=args.n_samples,
        cfg_scale=args.cfg_scale,
        dim=args.dim
    )
    
    print("\nFinal Results:")
    print(f"Wavelet Coherence: {results['wavelet_coherence']:.6f}")
    print(f"Discriminative Score: {results['discriminative_score']:.6f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.6f}")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.6f}")

