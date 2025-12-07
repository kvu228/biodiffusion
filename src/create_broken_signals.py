"""
Script to create broken signals from MIT-BIH test data for testing Signal Conditional model.
Creates 5 files, one for each class, with different types of broken signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from MITBIH import reverse_cls_dit
except ImportError:
    # Fallback if MITBIH module not available
    reverse_cls_dit = {
        0: 'Non-Ectopic Beats',
        1: 'Superventrical Ectopic', 
        2: 'Ventricular Beats',
        3: 'Unknown',
        4: 'Fusion Beats'
    }

def create_broken_signals(test_file_path, output_dir, num_samples_per_class=1):
    """
    Create broken signals from MIT-BIH test data.
    
    Args:
        test_file_path: Path to mitbih_test.csv
        output_dir: Directory to save broken signal files
        num_samples_per_class: Number of samples to create per class
    """
    # Read test data using pandas
    print(f"Loading test data from {test_file_path}...")
    data_pd = pd.read_csv(test_file_path, header=None)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_id in range(5):
        class_name = reverse_cls_dit[class_id]
        print(f"\nProcessing Class {class_id}: {class_name}")
        
        # Get samples for this class (label is in column 187)
        class_data = data_pd[data_pd[187] == class_id]
        
        if len(class_data) == 0:
            print(f"  No samples found for class {class_id}")
            continue
        
        # Take first num_samples_per_class samples
        samples = class_data.iloc[:num_samples_per_class]
        
        # Extract signal values (columns 0-127)
        signals = samples.iloc[:, :128].values
        
        # Create different types of broken signals
        broken_signals_list = []
        
        for idx, signal in enumerate(signals):
            # Original signal (for reference)
            original = signal.copy()
            
            # 1. Noisy signal (thermal noise + motion artifacts)
            noisy_signal = original.copy()
            # Add thermal noise (white noise)
            thermal_noise = np.random.randn(128) * 0.1
            noisy_signal += thermal_noise
            # Add motion artifacts (random spikes)
            num_spikes = np.random.randint(1, 4)
            spike_positions = np.random.randint(0, 128, num_spikes)
            for pos in spike_positions:
                spike_amp = np.random.uniform(0.3, 0.8)
                noisy_signal[pos:min(pos+5, 128)] += spike_amp
            
            # 2. Masked signal (missing values set to 0)
            masked_signal = original.copy()
            mask_ratio = 0.3
            mask_indices = np.random.choice(128, size=int(128 * mask_ratio), replace=False)
            masked_signal[mask_indices] = 0
            
            # 3. Downsampled signal (downsample then upsample)
            downsample_factor = 4
            downsampled = original[::downsample_factor]
            # Upsample using numpy interpolation (no scipy dependency)
            x_old = np.arange(len(downsampled))
            x_new = np.linspace(0, len(downsampled)-1, 128)
            # Simple linear interpolation
            downsampled_signal = np.interp(x_new, x_old, downsampled)
            
            # Store broken signals
            broken_signals_list.append({
                'original': original,
                'noisy': noisy_signal,
                'masked': masked_signal,
                'downsampled': downsampled_signal
            })
        
        # Save each type of broken signal
        for broken_type in ['noisy', 'masked', 'downsampled']:
            # Create array with all broken signals of this type
            broken_array = np.array([sig[broken_type] for sig in broken_signals_list])
            
            # If multiple samples, take first one for simplicity
            if broken_array.ndim == 2 and broken_array.shape[0] > 1:
                broken_array = broken_array[0]
            
            # Ensure it's 1D
            if broken_array.ndim == 2:
                broken_array = broken_array.flatten()
            
            # Save to CSV
            output_file = output_dir / f"mitbih_test_class{class_id}_{broken_type}.csv"
            np.savetxt(output_file, broken_array, delimiter=',', fmt='%.18e')
            print(f"  Saved {broken_type} signal to {output_file}")
            
            # Also save original for comparison
            original_array = np.array([sig['original'] for sig in broken_signals_list])
            if original_array.ndim == 2 and original_array.shape[0] > 1:
                original_array = original_array[0]
            if original_array.ndim == 2:
                original_array = original_array.flatten()
            original_file = output_dir / f"mitbih_test_class{class_id}_original.csv"
            np.savetxt(original_file, original_array, delimiter=',', fmt='%.18e')
            print(f"  Saved original signal to {original_file}")

if __name__ == '__main__':
    # Paths
    test_file = Path(__file__).parent / "datasets" / "heartbeat" / "mitbih_test.csv"
    output_dir = Path(__file__).parent / "datasets" / "heartbeat" / "broken_signals"
    
    print("Creating broken signals from MIT-BIH test data...")
    print(f"Test file: {test_file}")
    print(f"Output directory: {output_dir}")
    
    create_broken_signals(test_file, output_dir, num_samples_per_class=1)
    
    print("\n✅ Done! Created broken signal files:")
    print(f"   Location: {output_dir}")
    print("\nFiles created:")
    print("  - mitbih_test_class0_original.csv, noisy.csv, masked.csv, downsampled.csv")
    print("  - mitbih_test_class1_original.csv, noisy.csv, masked.csv, downsampled.csv")
    print("  - mitbih_test_class2_original.csv, noisy.csv, masked.csv, downsampled.csv")
    print("  - mitbih_test_class3_original.csv, noisy.csv, masked.csv, downsampled.csv")
    print("  - mitbih_test_class4_original.csv, noisy.csv, masked.csv, downsampled.csv")

