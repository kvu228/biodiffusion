"""1D Signal generation page with classifier-free guidance"""

import streamlit as st
import torch
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.core.model_loader import ModelLoader
    from streamlit_app.components.signal_display import display_signals
except ImportError:
    from core.model_loader import ModelLoader
    from streamlit_app.components.signal_display import display_signals

st.set_page_config(page_title="Signal Generation", page_icon="📈", layout="wide")

st.title("📈 1D Signal Generation")
st.markdown("Generate ECG/heartbeat signals using classifier-free guidance diffusion models")

# Instructions/Help section
with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
    st.markdown("""
    ### Cách sử dụng:
    
    1. **Chọn Model**: Chọn model đã được train từ danh sách trong sidebar
    2. **Cấu hình Model Parameters**:
       - **Number of classes**: Số lượng classes mà model đã được train (ví dụ: 5 cho MIT-BIH dataset)
       - **Sequence length**: Độ dài của signal (phải khớp với model đã train, thường là 128)
       - **Model dimension**: Dimension của model (thường là 64)
    
    3. **Load Model**: Click nút "🔄 Reload Model" để load model vào memory
    
    4. **Cấu hình Generation Parameters**:
       - **Number of samples**: Số lượng signals muốn generate (1-16)
       - **CFG Scale**: Classifier-free guidance scale (0.0-10.0)
         - Giá trị cao hơn = signal gần với class hơn nhưng có thể kém đa dạng
         - Giá trị thấp hơn = signal đa dạng hơn nhưng có thể ít phù hợp với class
         - Khuyến nghị: 3.0-5.0
    
    5. **Chọn Class**:
       - **Single class**: Generate tất cả signals cùng một class
       - **Multiple classes**: Generate signals với nhiều classes khác nhau
       - **Random**: Generate signals với classes ngẫu nhiên
    
    6. **Generate**: Click "Generate Signals" để tạo signals
    
    ### Giải thích Loss:
    - **Loss** được tính bằng cách đánh giá generated signals qua model
    - Loss thấp = signal được model đánh giá là tốt, phù hợp với distribution đã học
    - Loss cao = signal có thể không phù hợp hoặc model chưa học tốt
    - Loss chỉ là một chỉ số tham khảo, không phải là tiêu chuẩn duy nhất để đánh giá chất lượng
    """)

# Sidebar configuration
st.sidebar.header("Configuration")

device = st.sidebar.selectbox(
    "Device",
    ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
)

# Model loading - check signal checkpoint directory
checkpoint_dir = project_root / "src" / "signal" / "checkpoint"

available_models = {}
# Check signal checkpoint directory
if checkpoint_dir.exists():
    signal_models = ModelLoader.list_available_1d_models(str(checkpoint_dir))
    available_models.update(signal_models)

if not available_models:
    st.error("No models found. Please place trained models in src/signal/checkpoint/<run_name>/checkpoint.pt")
    st.stop()

model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_info = available_models[model_name]

# Model parameters
st.sidebar.subheader("Model Parameters")
st.sidebar.markdown("*Các tham số này phải khớp với model đã được train*")

num_classes = st.sidebar.number_input(
    "Number of classes", 
    min_value=1, 
    max_value=100, 
    value=5, 
    step=1,
    help="Số lượng classes mà model đã được train (ví dụ: 5 cho MIT-BIH dataset)"
)
seq_length = st.sidebar.number_input(
    "Sequence length", 
    min_value=32, 
    max_value=512, 
    value=128, 
    step=32,
    help="Độ dài của signal (phải khớp với model đã train, thường là 128)"
)
dim = st.sidebar.number_input(
    "Model dimension", 
    min_value=16, 
    max_value=256, 
    value=64, 
    step=16,
    help="Dimension của model (thường là 64)"
)

# Load model button
model_key = f"signal_model_{model_name}_{device}"
reload_button = st.sidebar.button("🔄 Reload Model", type="secondary")

# Load model if not in session state or if reload button clicked
if model_key not in st.session_state or reload_button:
    with st.spinner("Loading model..."):
        try:
            checkpoint_path = model_info.get("checkpoint_path") or model_info.get("path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                model, diffusion = ModelLoader.load_1d_cls_free_model(
                    checkpoint_path,
                    num_classes=num_classes,
                    seq_length=seq_length,
                    device=device,
                    dim=dim
                )
                st.session_state[model_key] = (model, diffusion)
                st.session_state['signal_model'] = model
                st.session_state['signal_diffusion'] = diffusion
                st.session_state['signal_device'] = device
                st.session_state['signal_num_classes'] = num_classes
                if not reload_button:
                    st.success("Model loaded successfully!")
                else:
                    st.success("Model reloaded!")
            else:
                st.error(f"Checkpoint not found at {checkpoint_path}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Generation parameters
if 'signal_model' in st.session_state:
    st.subheader("Generation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider(
            "Number of samples", 
            1, 16, 4,
            help="Số lượng signals muốn generate cùng lúc"
        )
    with col2:
        cfg_scale = st.slider(
            "CFG Scale", 
            0.0, 10.0, 3.0, 
            step=0.5,
            help="Classifier-free guidance scale. Giá trị cao = signal gần với class hơn. Khuyến nghị: 3.0-5.0"
        )
    
    # Class selection
    st.subheader("Class Selection")
    st.markdown("*Chọn cách thức gán class cho signals được generate*")
    class_selection = st.radio(
        "Select class generation mode",
        ["Single class", "Multiple classes", "Random"],
        help="Single class: tất cả signals cùng class | Multiple classes: chọn nhiều classes | Random: classes ngẫu nhiên"
    )
    
    labels = None
    if class_selection == "Single class":
        class_label = st.number_input(
            "Class label",
            min_value=0,
            max_value=st.session_state.get('signal_num_classes', num_classes) - 1,
            value=0,
            step=1
        )
        labels = torch.tensor([class_label] * n_samples, device=st.session_state['signal_device'])
    elif class_selection == "Multiple classes":
        num_classes_sel = st.session_state.get('signal_num_classes', num_classes)
        selected_classes = st.multiselect(
            "Select classes",
            options=list(range(num_classes_sel)),
            default=[0]
        )
        if selected_classes:
            # Repeat selected classes to match n_samples
            labels = torch.tensor([selected_classes[i % len(selected_classes)] for i in range(n_samples)], 
                                device=st.session_state['signal_device'])
        else:
            labels = torch.tensor([0] * n_samples, device=st.session_state['signal_device'])
    else:
        # Random classes
        labels = torch.randint(0, st.session_state.get('signal_num_classes', num_classes), 
                              (n_samples,), device=st.session_state['signal_device'])
    
    if st.button("Generate Signals", type="primary"):
        with st.spinner("Generating signals..."):
            try:
                diffusion = st.session_state['signal_diffusion']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Note: The sample method in GaussianDiffusion1D_cls_free takes classes tensor
                # and returns signals. We need to ensure labels are properly formatted.
                with torch.no_grad():
                    signals = diffusion.sample(
                        classes=labels,
                        cond_scale=cfg_scale
                    )
                
                progress_bar.progress(1.0)
                status_text.text("Generation complete!")
                
                # Calculate loss for generated signals
                with st.spinner("Calculating loss..."):
                    try:
                        # Signals from sample() are in [0, 1] range, need to normalize to [-1, 1] for loss calculation
                        # normalize_to_neg_one_to_one: img * 2 - 1
                        signals_normalized = signals * 2 - 1
                        
                        # Calculate loss using forward pass
                        loss = diffusion(signals_normalized, classes=labels)
                        loss_value = loss.item()
                        
                        # Display loss
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Loss", f"{loss_value:.6f}")
                        with col2:
                            st.metric("Number of Signals", n_samples)
                        with col3:
                            st.metric("CFG Scale", f"{cfg_scale:.1f}")
                        
                        # Loss interpretation
                        if loss_value < 0.1:
                            st.success(f"✅ Loss thấp ({loss_value:.6f}) - Signals có chất lượng tốt!")
                        elif loss_value < 0.5:
                            st.info(f"ℹ️ Loss trung bình ({loss_value:.6f}) - Signals có chất lượng ổn định")
                        else:
                            st.warning(f"⚠️ Loss cao ({loss_value:.6f}) - Có thể cần điều chỉnh CFG scale hoặc kiểm tra model")
                            
                    except Exception as e:
                        st.warning(f"Không thể tính loss: {str(e)}")
                        loss_value = None
                
                st.subheader("Generated Signals")
                display_signals(signals, labels=labels.cpu().numpy(), n_cols=min(n_samples, 4))
                
                # Additional info
                st.markdown("---")
                st.markdown("### 📊 Thông tin Signals")
                st.info(f"""
                - **Shape**: {list(signals.shape)} (batch, channels, sequence_length)
                - **Device**: {signals.device}
                - **Data type**: {signals.dtype}
                - **Value range**: [{signals.min().item():.3f}, {signals.max().item():.3f}]
                """)
                
            except Exception as e:
                st.error(f"Error generating signals: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("Please load a model from the sidebar to start generating signals.")

