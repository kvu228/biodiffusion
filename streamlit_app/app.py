"""Main Streamlit application"""

import streamlit as st
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="BioDiffusion Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧬 BioDiffusion Signal Demo")
    st.markdown("### 1D Signal Generation with Diffusion Models")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the **BioDiffusion Signal Demo**! 🎉
    
    Ứng dụng này cho phép bạn generate và đánh giá chất lượng 1D signals (ECG, heartbeat, etc.) sử dụng 
    diffusion models với classifier-free guidance.
    
    ### 🚀 Bắt đầu nhanh
    
    1. **Xem Dataset** (trang "Dataset"): Hiểu về dữ liệu được sử dụng để train
    2. **Generate Signals** (trang "Signal Generation"):
       - Chọn model đã được train từ danh sách
       - Cấu hình tham số model (số classes, độ dài sequence, dimension)
       - Load model bằng nút "🔄 Reload Model"
       - Chọn tham số generation (số lượng samples, CFG scale)
       - Chọn class cho signals
       - Click "Generate Signals" để tạo signals
    
    ### 📁 Yêu cầu Model
    
    Đặt các model đã train vào thư mục `src/signal/checkpoint/`:
    - `src/signal/checkpoint/<run_name>/checkpoint.pt`
    
    Ví dụ: `src/signal/checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt`
    
    ### ✨ Tính năng chính
    
    - ✅ **Dataset Information**: Xem thông tin và samples từ MIT-BIH dataset
    - ✅ **Generate 1D Signals**: Tạo ECG/heartbeat signals với class conditioning
    - ✅ **Tính toán Loss**: Đánh giá chất lượng signals được generate
    - ✅ **Chọn Class linh hoạt**: Single class, multiple classes, hoặc random
    - ✅ **Điều chỉnh CFG Scale**: Kiểm soát độ mạnh của classifier-free guidance
    - ✅ **Visualization tương tác**: Xem signals trong grid layout
    - ✅ **Hướng dẫn chi tiết**: Tooltips và hướng dẫn trong UI
    
    ### 💡 Mẹo sử dụng
    
    - **CFG Scale**: Giá trị 3.0-5.0 thường cho kết quả tốt
    - **Loss**: Loss thấp (< 0.1) = signals chất lượng tốt
    - **Model Parameters**: Phải khớp với model đã train
    - Xem hướng dẫn chi tiết trong trang Signal Generation (icon 📖)
    """)
    
    # Device info
    st.sidebar.header("System Information")
    if torch.cuda.is_available():
        st.sidebar.success(f"CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("CUDA not available. Using CPU.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("Use the pages menu above to navigate to signal generation.")

if __name__ == "__main__":
    main()

