"""Dataset information and visualization page"""

import streamlit as st
import torch
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.signal.MITBIH import mitbih_allClass, reverse_cls_dit, cls_dit
    from streamlit_app.components.signal_display import display_signals
except ImportError:
    try:
        from signal.MITBIH import mitbih_allClass, reverse_cls_dit, cls_dit
        from streamlit_app.components.signal_display import display_signals
    except ImportError:
        st.error("Cannot import MITBIH dataset. Please check if dataset files are available.")
        st.stop()

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

st.title("📊 MIT-BIH Arrhythmia Dataset")
st.markdown("Giới thiệu và visualization của dataset được sử dụng để train model")

# Dataset information
st.header("📖 Thông tin về Dataset")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### MIT-BIH Arrhythmia Database
    
    **Mô tả:**
    - Dataset ECG (Electrocardiogram) được sử dụng rộng rãi trong nghiên cứu về nhịp tim bất thường
    - Bao gồm các heartbeat signals được gán nhãn thành 5 classes khác nhau
    - Mỗi signal có độ dài 128 timesteps
    
    **Nguồn:**
    - MIT-BIH Arrhythmia Database từ PhysioNet
    - Được sử dụng trong nhiều nghiên cứu về ECG classification
    """)

with col2:
    st.markdown("""
    ### 5 Classes trong Dataset
    
    1. **Non-Ectopic Beats (Class 0)**: Nhịp tim bình thường
    2. **Superventrical Ectopic (Class 1)**: Nhịp tim bất thường từ tâm nhĩ
    3. **Ventricular Beats (Class 2)**: Nhịp tim bất thường từ tâm thất
    4. **Unknown (Class 3)**: Nhịp tim không xác định
    5. **Fusion Beats (Class 4)**: Nhịp tim kết hợp
    
    **Đặc điểm:**
    - Sequence length: 128 timesteps
    - Channels: 1 (single channel ECG)
    - Format: Normalized values
    """)

st.markdown("---")

# Load dataset
st.header("📈 Dataset Statistics & Visualization")

dataset_path = project_root / "src" / "signal" / "datasets" / "heartbeat" / "mitbih_train.csv"
if not dataset_path.exists():
    # Try alternative path
    dataset_path = project_root / "src" / "signal" / "datasets" / "mitbih_train.csv"
if not dataset_path.exists():
    # Try another alternative
    dataset_path = project_root / "datasets" / "heartbeat" / "mitbih_train.csv"

if dataset_path.exists():
    try:
        # Load dataset
        with st.spinner("Loading dataset..."):
            dataset = mitbih_allClass(filename=str(dataset_path), isBalanced=False)
            
        # Statistics
        st.subheader("📊 Thống kê Dataset")
        
        # Count samples per class
        class_counts = {}
        for i in range(5):
            class_name = reverse_cls_dit[i]
            # Get count from dataset
            if hasattr(dataset, 'data_0'):
                if i == 0:
                    class_counts[class_name] = len(dataset.data_0)
                elif i == 1:
                    class_counts[class_name] = len(dataset.data_1)
                elif i == 2:
                    class_counts[class_name] = len(dataset.data_2)
                elif i == 3:
                    class_counts[class_name] = len(dataset.data_3)
                elif i == 4:
                    class_counts[class_name] = len(dataset.data_4)
            else:
                # Fallback: count from labels
                labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
                class_counts[class_name] = labels.count(i)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        total_samples = sum(class_counts.values())
        
        with col1:
            st.metric("Total Samples", f"{total_samples:,}")
        with col2:
            st.metric("Number of Classes", "5")
        with col3:
            st.metric("Sequence Length", "128")
        
        # Class distribution chart
        st.subheader("📊 Phân bố Classes")
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        bars = ax.bar(classes, counts, color=colors)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("Class Distribution in MIT-BIH Dataset", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display class distribution as table
        st.markdown("#### Chi tiết phân bố:")
        df_stats = pd.DataFrame({
            'Class': classes,
            'Label': [cls_dit[c] for c in classes],
            'Samples': counts,
            'Percentage': [f"{(c/total_samples)*100:.2f}%" for c in counts]
        })
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Sample visualization
        st.subheader("🎨 Sample Signals từ mỗi Class")
        st.markdown("*Hiển thị một số samples đại diện từ mỗi class để hiểu rõ hơn về dữ liệu*")
        
        # Number of samples to show per class
        n_samples_per_class = st.slider(
            "Số samples hiển thị mỗi class",
            min_value=1,
            max_value=5,
            value=3,
            help="Chọn số lượng samples muốn xem từ mỗi class"
        )
        
        if st.button("🔄 Load Samples", type="primary"):
            with st.spinner("Loading samples..."):
                try:
                    # Collect samples from each class
                    all_samples = []
                    all_labels = []
                    
                    for class_id in range(5):
                        class_name = reverse_cls_dit[class_id]
                        samples_collected = 0
                        
                        # Try to get samples from dataset
                        for idx in range(len(dataset)):
                            signal, label = dataset[idx]
                            if label == class_id and samples_collected < n_samples_per_class:
                                # Convert to tensor if needed
                                if isinstance(signal, np.ndarray):
                                    signal = torch.from_numpy(signal).float()
                                
                                # Ensure signal has shape (1, seq_length) for display
                                if signal.dim() == 1:
                                    signal = signal.unsqueeze(0)
                                elif signal.dim() == 3:
                                    signal = signal.squeeze(0)  # Remove extra dimension
                                
                                all_samples.append(signal)
                                all_labels.append(int(label))
                                samples_collected += 1
                                
                                if samples_collected >= n_samples_per_class:
                                    break
                    
                    if all_samples:
                        # Stack all samples - shape should be (batch, channels, seq_length)
                        samples_tensor = torch.stack(all_samples, dim=0)
                        # Ensure shape is (batch, channels, seq_length)
                        if samples_tensor.dim() == 2:
                            samples_tensor = samples_tensor.unsqueeze(1)
                        labels_array = np.array(all_labels)
                        
                        st.success(f"✅ Đã load {len(samples_tensor)} samples!")
                        
                        # Display samples
                        st.markdown("### 📈 Visualization")
                        display_signals(samples_tensor, labels=labels_array, n_cols=5, figsize=(20, 12))
                        
                        # Class information
                        st.markdown("### 📋 Thông tin Samples")
                        for class_id in range(5):
                            class_name = reverse_cls_dit[class_id]
                            class_samples = [i for i, lbl in enumerate(labels_array) if lbl == class_id]
                            if class_samples:
                                st.markdown(f"**{class_name} (Class {class_id})**: {len(class_samples)} samples")
                    else:
                        st.warning("Không thể load samples từ dataset. Vui lòng kiểm tra lại dataset path.")
                        
                except Exception as e:
                    st.error(f"Error loading samples: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
        
        # Dataset characteristics
        st.markdown("---")
        st.subheader("🔍 Đặc điểm Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Format:**
            - Shape: `(batch, channels, sequence_length)`
            - Channels: 1 (single channel ECG)
            - Sequence length: 128 timesteps
            - Data type: Float32
            - Value range: Normalized
            
            **Preprocessing:**
            - Signals được normalize về range [0, 1]
            - Mỗi signal đại diện cho một heartbeat
            - Timesteps được lấy mẫu từ ECG recording
            """)
        
        with col2:
            st.markdown("""
            **Usage:**
            - Training diffusion models với classifier-free guidance
            - Conditional generation theo class labels
            - Evaluation và testing model performance
            
            **Note:**
            - Dataset có thể không cân bằng giữa các classes
            - Một số classes có thể có ít samples hơn
            - Model được train để handle class imbalance
            """)
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info(f"Dataset path tried: {dataset_path}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
else:
    st.warning("⚠️ Dataset file not found!")
    st.markdown(f"""
    **Expected paths:**
    - `src/signal/datasets/heartbeat/mitbih_train.csv`
    - `src/signal/datasets/mitbih_train.csv`
    - `datasets/heartbeat/mitbih_train.csv`
    
    **Current path checked:** `{dataset_path}`
    
    Vui lòng đảm bảo dataset file tồn tại tại một trong các đường dẫn trên.
    """)
    
    # Show dataset information even if file not found
    st.markdown("---")
    st.subheader("📖 Thông tin Dataset (Reference)")
    
    st.markdown("""
    ### MIT-BIH Arrhythmia Database
    
    **Classes:**
    1. **Non-Ectopic Beats (0)**: Normal heartbeats
    2. **Superventrical Ectopic (1)**: Abnormal beats from atria
    3. **Ventricular Beats (2)**: Abnormal beats from ventricles
    4. **Unknown (3)**: Unclassified beats
    5. **Fusion Beats (4)**: Combined beats
    
    **Characteristics:**
    - Sequence length: 128 timesteps
    - Single channel ECG signals
    - Used for arrhythmia classification and generation
    """)

