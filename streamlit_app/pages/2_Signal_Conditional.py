"""Signal Conditional generation page - for denoising, imputation, super-resolution"""

import streamlit as st
import torch
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.model_loader import ModelLoader
    from streamlit_app.components.signal_display import display_signal_single
except ImportError:
    from model_loader import ModelLoader
    from streamlit_app.components.signal_display import display_signal_single

st.set_page_config(page_title="Signal Conditional", page_icon="🔧", layout="wide")

st.title("🔧 Signal Conditional Generation")
st.markdown("Restore broken signals using **Signal Conditional** diffusion models")
st.caption("Based on BioDiffusion paper: Signal conditional generation for denoising, imputation, super-resolution, and individual signal generation")

# Instructions/Help section
with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
    st.markdown("""
    ### Loại Model:
    
    Đây là **Signal Conditional Diffusion Model** (theo paper BioDiffusion):
    - Nhận một signal bị broken (noisy, masked, downsampled) làm input
    - Model sẽ restore/denoise/impute/super-resolve signal đó
    - Không tạo signal mới từ đầu, mà restore từ signal có sẵn
    
    ### Ứng dụng:
    
    1. **Signal Denoising**: Loại bỏ nhiễu từ signal bị nhiễu
       - Thermal noise (white noise)
       - Electrode contact noise (low-frequency drift)
       - Motion artifacts (random spikes)
    
    2. **Signal Imputation**: Điền các giá trị bị thiếu trong signal
    
    3. **Signal Super-resolution**: Tăng độ phân giải của signal
    
    4. **Individual Signal Generation**: Generate nhiều samples từ một signal của một subject
    
    ### Cách sử dụng:
    
    1. **Chọn Model**: Chọn signal conditional model đã được train từ danh sách trong sidebar
    2. **Cấu hình Model Parameters**:
       - **Sequence length**: Độ dài của signal (phải khớp với model đã train)
       - **Model dimension**: Dimension của model (thường là 64)
       - **Channels**: Số channels (1 cho MIT-BIH, 3 cho UNIMIB)
    
    3. **Load Model**: Click nút "🔄 Reload Model" để load model vào memory
    
    4. **Chọn Input Signal**:
       - **Upload signal**: Upload file CSV hoặc numpy array
       - **Generate broken signal**: Tạo signal bị broken từ dataset có sẵn
       - **Manual input**: Nhập signal thủ công
    
    5. **Chọn loại broken signal**:
       - **Noisy**: Thêm noise vào signal
       - **Masked**: Mask một phần signal (set về 0)
       - **Downsampled**: Giảm độ phân giải
    
    6. **Restore**: Click "Restore Signal" để restore signal bị broken
    """)

# Sidebar configuration
st.sidebar.header("Configuration")

device = st.sidebar.selectbox(
    "Device",
    ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
)

# Model loading - check signal checkpoint directory
checkpoint_dir = project_root / "src" / "checkpoint"

available_models = {}
# Check signal checkpoint directory
if checkpoint_dir.exists():
    signal_models = ModelLoader.list_available_1d_models(str(checkpoint_dir))
    available_models.update(signal_models)

if not available_models:
    st.error("No models found. Please place trained signal conditional models in src/checkpoint/<run_name>/checkpoint.pt")
    st.info("Note: Signal conditional models should be trained with modules1D_cond (Unet1D, GaussianDiffusion1D)")
    st.stop()

model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_info = available_models[model_name]

# Model parameters
st.sidebar.subheader("Model Parameters")
st.sidebar.markdown("*Các tham số này phải khớp với model đã được train*")

seq_length = st.sidebar.number_input(
    "Sequence length", 
    min_value=32, 
    max_value=512, 
    value=128, 
    step=32,
    help="Độ dài của signal (phải khớp với model đã train)"
)
dim = st.sidebar.number_input(
    "Model dimension", 
    min_value=16, 
    max_value=256, 
    value=64, 
    step=16,
    help="Dimension của model (thường là 64)"
)
channels = st.sidebar.number_input(
    "Channels", 
    min_value=1, 
    max_value=10, 
    value=1, 
    step=1,
    help="Số channels (1 cho MIT-BIH, 3 cho UNIMIB)"
)

# Load model button
model_key = f"signal_cond_model_{model_name}_{device}"
reload_button = st.sidebar.button("🔄 Reload Model", type="secondary")

# Load model if not in session state or if reload button clicked
if model_key not in st.session_state or reload_button:
    with st.spinner("Loading signal conditional model..."):
        try:
            checkpoint_path = model_info.get("checkpoint_path") or model_info.get("path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                model, diffusion = ModelLoader.load_1d_signal_cond_model(
                    checkpoint_path,
                    seq_length=seq_length,
                    device=device,
                    dim=dim,
                    channels=channels
                )
                st.session_state[model_key] = (model, diffusion)
                st.session_state['signal_cond_model'] = model
                st.session_state['signal_cond_diffusion'] = diffusion
                st.session_state['signal_cond_device'] = device
                st.session_state['signal_cond_seq_length'] = seq_length
                st.session_state['signal_cond_channels'] = channels
                if not reload_button:
                    st.success("Model loaded successfully!")
                else:
                    st.success("Model reloaded!")
            else:
                st.error(f"Checkpoint not found at {checkpoint_path}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# Main content
if 'signal_cond_model' in st.session_state:
    st.subheader("Input Signal Configuration")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method",
        ["Generate broken signal", "Upload signal file"],
        help="Chọn cách thức nhập signal bị broken"
    )
    
    input_signal = None
    original_signal = None
    
    if input_method == "Generate broken signal":
        st.info("💡 Generate a broken signal from dataset (for testing purposes)")
        # This would require dataset loading - simplified for now
        st.warning("Feature coming soon: Generate broken signal from dataset")
        
    elif input_method == "Upload signal file":
        uploaded_file = st.file_uploader("Upload signal file", type=['csv', 'npy', 'txt'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = np.loadtxt(uploaded_file, delimiter=',')
                elif uploaded_file.name.endswith('.npy'):
                    data = np.load(uploaded_file)
                else:
                    data = np.loadtxt(uploaded_file)
                
                # Ensure correct shape
                if data.ndim == 1:
                    data = data.reshape(1, -1)  # (1, seq_length)
                elif data.ndim == 2:
                    if data.shape[0] > data.shape[1]:
                        data = data.T  # Transpose if needed
                
                # Ensure correct length
                seq_len = st.session_state.get('signal_cond_seq_length', seq_length)
                if data.shape[1] > seq_len:
                    data = data[:, :seq_len]
                elif data.shape[1] < seq_len:
                    # Pad with zeros
                    pad_width = seq_len - data.shape[1]
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
                
                original_signal = torch.from_numpy(data).float().to(st.session_state['signal_cond_device'])
                st.success(f"Signal loaded: shape {original_signal.shape}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Broken signal type selection
    if original_signal is not None:
        st.subheader("Broken Signal Type")
        broken_type = st.selectbox(
            "Select type of broken signal",
            ["Noisy", "Masked", "Downsampled"],
            help="Chọn loại signal bị broken để test model"
        )
        
        broken_signal = original_signal.clone()
        
        if broken_type == "Noisy":
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.2, 0.1)
            noise = torch.randn_like(broken_signal) * noise_level
            broken_signal = broken_signal + noise
            st.info(f"Added noise with level {noise_level}")
        
        elif broken_type == "Masked":
            mask_ratio = st.slider("Mask ratio", 0.0, 1.0, 0.3, 0.1)
            mask_indices = torch.randperm(broken_signal.shape[1])[:int(broken_signal.shape[1] * mask_ratio)]
            broken_signal[:, mask_indices] = 0
            st.info(f"Masked {mask_ratio*100:.1f}% of signal")
        
        elif broken_type == "Downsampled":
            downsample_factor = st.slider("Downsample factor", 2, 8, 2, 1)
            # Downsample
            downsampled = broken_signal[:, ::downsample_factor]
            # Upsample back using simple interpolation
            from torch.nn import functional as F
            broken_signal = F.interpolate(
                downsampled.unsqueeze(0), 
                size=broken_signal.shape[1], 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            st.info(f"Downsampled by factor {downsample_factor} then upsampled")
        
        # Display original and broken signals
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Signal")
            display_signal_single(original_signal, label="Original")
        with col2:
            st.markdown("### Broken Signal")
            display_signal_single(broken_signal, label=broken_type)
        
        # Restore button
        if st.button("🔧 Restore Signal", type="primary"):
            with st.spinner("Restoring signal..."):
                try:
                    diffusion = st.session_state['signal_cond_diffusion']
                    
                    # Ensure correct shape: (batch, channels, seq_length)
                    # Model expects input_cond in same format as training
                    # According to code, input_cond is used directly without normalization
                    if broken_signal.dim() == 2:
                        if broken_signal.shape[0] == 1:
                            # (1, seq_length) -> (1, channels, seq_length)
                            broken_signal_reshaped = broken_signal.unsqueeze(1)
                        else:
                            # (channels, seq_length) -> (1, channels, seq_length)
                            broken_signal_reshaped = broken_signal.unsqueeze(0)
                    else:
                        broken_signal_reshaped = broken_signal
                    
                    # Ensure we have correct number of channels
                    num_channels = st.session_state.get('signal_cond_channels', channels)
                    if broken_signal_reshaped.shape[1] != num_channels:
                        # Repeat or slice to match channels
                        if broken_signal_reshaped.shape[1] == 1 and num_channels > 1:
                            broken_signal_reshaped = broken_signal_reshaped.repeat(1, num_channels, 1)
                        elif broken_signal_reshaped.shape[1] > num_channels:
                            broken_signal_reshaped = broken_signal_reshaped[:, :num_channels, :]
                    
                    # Sample with conditional input
                    # input_cond is used as initial x_start in self-conditioning
                    with torch.no_grad():
                        restored_signal = diffusion.sample(
                            input_cond=broken_signal_reshaped,
                            batch_size=1
                        )
                    
                    st.success("Signal restored!")
                    
                    # Ensure restored_signal has same shape as original_signal for display and metrics
                    # restored_signal might be (batch, channels, seq_length), squeeze batch if needed
                    if restored_signal.dim() == 3 and restored_signal.shape[0] == 1:
                        restored_signal_display = restored_signal.squeeze(0)  # (1, channels, seq_length) -> (channels, seq_length)
                    elif restored_signal.dim() == 3:
                        restored_signal_display = restored_signal[0]  # (batch, channels, seq_length) -> (channels, seq_length)
                    else:
                        restored_signal_display = restored_signal
                    
                    # Ensure original_signal has same shape for comparison
                    if original_signal.dim() == 3 and original_signal.shape[0] == 1:
                        original_signal_display = original_signal.squeeze(0)
                    elif original_signal.dim() == 3:
                        original_signal_display = original_signal[0]
                    else:
                        original_signal_display = original_signal
                    
                    # Display results
                    st.subheader("Restoration Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("### Original")
                        display_signal_single(original_signal_display, label="Original")
                    with col2:
                        st.markdown("### Broken")
                        display_signal_single(broken_signal, label=broken_type)
                    with col3:
                        st.markdown("### Restored")
                        display_signal_single(restored_signal_display, label="Restored")
                    
                    # Calculate metrics
                    st.markdown("---")
                    st.subheader("📊 Restoration Metrics")
                    
                    # Ensure shapes match for metrics calculation
                    if original_signal_display.shape != restored_signal_display.shape:
                        # Try to match shapes
                        if original_signal_display.dim() == 2 and restored_signal_display.dim() == 2:
                            # Both are 2D, ensure same shape
                            min_channels = min(original_signal_display.shape[0], restored_signal_display.shape[0])
                            min_length = min(original_signal_display.shape[1], restored_signal_display.shape[1])
                            original_signal_display = original_signal_display[:min_channels, :min_length]
                            restored_signal_display = restored_signal_display[:min_channels, :min_length]
                    
                    # MSE between original and restored
                    mse = torch.mean((original_signal_display - restored_signal_display) ** 2).item()
                    mae = torch.mean(torch.abs(original_signal_display - restored_signal_display)).item()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                    with col2:
                        st.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
                    
                    if mse < 0.01:
                        st.success("✅ Excellent restoration! MSE < 0.01")
                    elif mse < 0.1:
                        st.info("ℹ️ Good restoration! MSE < 0.1")
                    else:
                        st.warning("⚠️ Restoration may need improvement. Consider adjusting model or input signal.")
                    
                    # Additional info
                    st.markdown("---")
                    st.markdown("### 📋 Signal Information")
                    st.info(f"""
                    - **Original shape**: {list(original_signal_display.shape)}
                    - **Restored shape**: {list(restored_signal_display.shape)}
                    - **Device**: {restored_signal_display.device if isinstance(restored_signal_display, torch.Tensor) else 'numpy'}
                    - **Data type**: {restored_signal_display.dtype}
                    - **Value range (restored)**: [{restored_signal_display.min().item() if isinstance(restored_signal_display, torch.Tensor) else restored_signal_display.min():.3f}, {restored_signal_display.max().item() if isinstance(restored_signal_display, torch.Tensor) else restored_signal_display.max():.3f}]
                    """)
                    
                except Exception as e:
                    st.error(f"Error restoring signal: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
    
    else:
        st.info("Please provide an input signal to restore.")
        
else:
    st.info("Please load a signal conditional model from the sidebar to start restoring signals.")

