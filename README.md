# BioDiffusion

A versatile diffusion model framework for biomedical signal synthesis, specifically designed for generating 1D ECG/heartbeat signals using Deep Diffusion Probabilistic Models (DDPM) with classifier-free guidance.

## Overview

BioDiffusion implements state-of-the-art diffusion models for generating biomedical signals, particularly focusing on ECG (Electrocardiogram) signals from the MIT-BIH Arrhythmia Database. The framework supports multiple generation modes:

- **Class-Conditional Generation**: Generate signals conditioned on class labels (5 classes: Non-Ectopic, Superventrical Ectopic, Ventricular Beats, Unknown, Fusion Beats)
- **Signal Conditional Generation**: Generate signals conditioned on other signals (for denoising, imputation, super-resolution)
- **Unconditional Generation**: Generate signals without any conditioning

## Features

- 🧬 **Diffusion Models**: Implementation of DDPM with classifier-free guidance
- 📊 **MIT-BIH Dataset**: Support for MIT-BIH Arrhythmia Database with 5 classes
- 🎯 **Class-Conditional Generation**: Generate signals for specific heartbeat classes
- 🔧 **Signal Processing**: Support for denoising, imputation, and super-resolution tasks
- 📈 **Streamlit Web App**: Interactive web interface for visualization and signal generation
- 📉 **Model Evaluation**: Comprehensive testing with metrics (Wavelet Coherence, Discriminative Score, F1-Score)
- 💾 **Checkpoint Management**: Automatic checkpoint saving and resuming with early stopping
- 📝 **TensorBoard Logging**: Training progress visualization

## Project Structure

```
biodiffusion/
├── src/
│   ├── ddpm1d_cls_free.py          # Class-conditional training script
│   ├── ddpm1d_sign_cond.py          # Signal-conditional training script
│   ├── diffusion1D.py               # Base diffusion model implementation
│   ├── Unet1D.py                     # UNet architecture for 1D signals
│   ├── MITBIH.py                     # MIT-BIH dataset loaders
│   ├── load_dataset.py               # Dataset download utilities
│   ├── test_model.py                 # Model testing and evaluation
│   ├── utils.py                      # Utility functions
│   ├── utils_cond.py                 # Conditional model utilities
│   ├── modules/
│   │   ├── modules1D_cls_free.py     # Classifier-free guidance modules
│   │   └── modules1D_cond.py         # Conditional diffusion modules
│   ├── datasets/
│   │   └── heartbeat/                # MIT-BIH dataset files
│   ├── checkpoint/                   # Model checkpoints
│   ├── results/                      # Generated samples
│   ├── logs/                         # Training logs
│   └── runs/                         # TensorBoard logs
├── streamlit_app/
│   ├── app.py                        # Main Streamlit app
│   ├── pages/
│   │   ├── 0_Dataset.py             # Dataset visualization page
│   │   └── 1_Signal_Generation.py    # Signal generation page
│   └── components/
│       └── signal_display.py         # Signal visualization components
├── docs/
│   └── report/                       # Documentation and theory
├── pyproject.toml                     # Project dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python >= 3.12
- CUDA-capable GPU (recommended) or CPU
- Kaggle API credentials (for dataset download)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd biodiffusion
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

3. Download the MIT-BIH dataset:
```bash
# Set up Kaggle API credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Or run the download script
python src/load_dataset.py
```

The dataset will be downloaded to `src/datasets/heartbeat/`.

## Usage

### Training

#### Class-Conditional Model (Classifier-Free Guidance)

Train a model to generate signals conditioned on class labels:

```bash
python src/ddpm1d_cls_free.py
```

Or customize training parameters:

```python
from src.ddpm1d_cls_free import train

train(
    run_name='DDPM1D_cls_free_MITBIH',
    epochs=300,
    batch_size=64,
    seq_length=128,
    num_classes=5,
    lr=1e-4,
    device='cuda',
    early_stopping=True,
    patience=20
)
```

#### Signal-Conditional Model

Train a model for signal conditioning tasks (denoising, imputation):

```bash
python src/ddpm1d_sign_cond.py
```

### Model Testing

Evaluate a trained model:

```bash
python src/test_model.py \
    --checkpoint src/checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt \
    --test_data src/datasets/heartbeat/mitbih_test.csv \
    --num_classes 5 \
    --seq_length 128 \
    --n_samples 100 \
    --cfg_scale 3.0
```

The script calculates:
- **Wavelet Coherence Score**: Measures spectral similarity between real and generated signals
- **Discriminative Score**: Measures how distinguishable generated signals are from real ones
- **F1-Score**: Classification performance on generated signals

### Streamlit Web App

Launch the interactive web interface:

```bash
streamlit run streamlit_app/app.py
```

The app provides:
- **Dataset Visualization**: Explore the MIT-BIH dataset with statistics and sample signals
- **Signal Generation**: Generate signals interactively with customizable parameters
- **Model Management**: Load and switch between different trained models

## Model Architecture

### UNet1D with Classifier-Free Guidance

The class-conditional model uses a 1D UNet architecture with:
- **Base dimension**: 64
- **Channel multipliers**: (1, 2, 4, 8)
- **Residual blocks**: Group normalization with 8 groups
- **Attention**: Multi-head attention (4 heads)
- **Class embeddings**: Learnable embeddings for each class
- **Conditional dropout**: 0.5 probability for classifier-free guidance

### Diffusion Process

- **Timesteps**: 1000
- **Noise schedule**: Cosine schedule
- **Objective**: Noise prediction (ε-prediction)
- **Loss**: L1 loss
- **Sampling**: DDPM sampling with optional DDIM acceleration

## Dataset

### MIT-BIH Arrhythmia Database

The framework uses the MIT-BIH Arrhythmia Database with 5 classes:

1. **Non-Ectopic Beats (Class 0)**: Normal heartbeats
2. **Superventrical Ectopic (Class 1)**: Abnormal beats from atria
3. **Ventricular Beats (Class 2)**: Abnormal beats from ventricles
4. **Unknown (Class 3)**: Unclassified beats
5. **Fusion Beats (Class 4)**: Combined beats

**Dataset Characteristics**:
- Sequence length: 128 timesteps
- Channels: 1 (single channel ECG)
- Format: Normalized values
- Training set: `mitbih_train.csv`
- Test set: `mitbih_test.csv`

## Training Configuration

### Default Training Parameters

**Class-Conditional Model**:
- Epochs: 300
- Batch size: 64
- Learning rate: 1e-4
- Optimizer: AdamW
- Early stopping: Enabled (patience=20)
- Checkpoint saving: Every epoch

**Signal-Conditional Model**:
- Epochs: 100
- Batch size: 32
- Learning rate: 3e-4
- Objective: v-parameterization
- Self-conditioning: Enabled

### Checkpoint Management

Models are automatically saved to `src/checkpoint/<run_name>/`:
- `checkpoint.pt`: Latest checkpoint
- `checkpoint_best.pt`: Best model (lowest loss)

Training can be resumed from checkpoints automatically.

## Results

Generated samples are saved to `src/results/<run_name>/` as image files showing:
- Signal waveforms for each class
- Class labels
- Grid layout for easy comparison

## Evaluation Metrics

The framework provides comprehensive evaluation:

1. **Wavelet Coherence Score**: Measures frequency-domain similarity
2. **Discriminative Score**: Measures how realistic generated signals are (lower is better)
3. **F1-Score**: Classification accuracy on generated signals

## Dependencies

Key dependencies:
- `torch >= 2.9.1`: Deep learning framework
- `einops >= 0.8.1`: Tensor operations
- `matplotlib >= 3.10.7`: Visualization
- `streamlit >= 1.52.1`: Web interface
- `tensorboard >= 2.20.0`: Training visualization
- `scikit-learn >= 1.7.2`: Evaluation metrics
- `numpy >= 2.3.5`: Numerical operations

See `pyproject.toml` for the complete list.

## Documentation

Detailed documentation is available in `docs/report/`:
- `biodiffusion_models_theory.md`: Theoretical background and model architectures
- `unet_architecture_detailed.md`: Detailed UNet architecture documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{biodiffusion2024,
  title={BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis},
  author={...},
  journal={arXiv preprint arXiv:2401.10282},
  year={2024}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MIT-BIH Arrhythmia Database from PhysioNet
- Based on diffusion model implementations from [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

## Troubleshooting

### Common Issues

1. **DataLoader hanging on Windows**: Set `num_workers=0` in training scripts
2. **CUDA out of memory**: Reduce batch size or sequence length
3. **Dataset not found**: Ensure dataset files are in `src/datasets/heartbeat/`
4. **Model checkpoint not loading**: Check that model parameters match training configuration

## Future Work

- [ ] Support for additional biomedical signal types
- [ ] Multi-channel signal generation
- [ ] Improved evaluation metrics
- [ ] Model compression and optimization
- [ ] Real-time generation capabilities

