# BioDiffusion: Lý Thuyết và Kiến Trúc của Ba Mô Hình Diffusion

## Tổng Quan

BioDiffusion là một framework diffusion-based được thiết kế đặc biệt cho việc tổng hợp tín hiệu y sinh học đa biến. Framework này hỗ trợ ba loại mô hình generation chính:

1. **Unconditional Diffusion Models** - Mô hình không điều kiện
2. **Label-Conditional Diffusion Models** - Mô hình điều kiện theo nhãn lớp (Classifier-Free Guidance)
3. **Signal Conditional Diffusion Models** - Mô hình điều kiện theo tín hiệu

---

## 1. Unconditional Diffusion Models

### 1.1 Lý Thuyết Cơ Bản

Mô hình Unconditional Diffusion là phiên bản cơ bản nhất của diffusion model, không sử dụng bất kỳ thông tin điều kiện nào trong quá trình generation. Mô hình này học phân phối dữ liệu thuần túy thông qua quá trình forward và reverse diffusion.

#### Forward Diffusion Process

Quá trình forward diffusion thêm nhiễu Gaussian vào dữ liệu gốc theo một lịch trình (schedule) được xác định trước:

\[
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
\]

Trong đó:
- \(x_0\) là tín hiệu gốc
- \(x_t\) là tín hiệu tại timestep \(t\)
- \(\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s\) với \(\alpha_s = 1 - \beta_s\)
- \(\beta_s\) là noise schedule (thường sử dụng cosine schedule)

#### Reverse Diffusion Process

Mô hình học cách đảo ngược quá trình forward diffusion bằng cách dự đoán nhiễu đã được thêm vào:

\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\]

Mục tiêu training là minimize loss:

\[
L = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]
\]

Trong đó \(\epsilon_\theta(x_t, t)\) là mạng neural network dự đoán nhiễu.

### 1.2 Kiến Trúc Mô Hình

Theo paper, mô hình Unconditional được train **riêng biệt cho mỗi class** trong dataset:

- **Architecture**:
  - Base channels: 64
  - Channel multipliers: (1, 2, 4, 8) cho UNIMIB và MITBIH
  - Channel multipliers: (1, 2, 4, 8, 8) cho Simulated Dataset
  - Residual blocks groups: 8
  - Attention heads: 4

- **Training Parameters**:
  - Optimizer: Adam
  - Batch size: 32
  - Learning rate: 3e-4
  - Epochs: 100
  - Loss: L1
  - Diffusion timesteps: 1000
  - Noise schedule: cosine

### 1.3 Implementation Notes

Trong codebase hiện tại, mô hình Unconditional có thể được implement bằng cách:
- Sử dụng `Unet1D` từ `modules1D_cond.py` với `self_condition=False`
- Hoặc sử dụng `Unet1D_cls_free` với `cond_drop_prob=1.0` (force unconditional)

**Lưu ý**: Hiện tại không có notebook riêng cho Unconditional model trong `src/signal/`, nhưng có thể sử dụng architecture tương tự như Signal Conditional model nhưng không có conditional input.

---

## 2. Label-Conditional Diffusion Models (Classifier-Free Guidance)

### 2.1 Lý Thuyết Classifier-Free Guidance

Mô hình Label-Conditional sử dụng kỹ thuật **Classifier-Free Guidance (CFG)** để điều khiển generation dựa trên class labels. Đây là một phương pháp hiệu quả hơn so với classifier guidance truyền thống vì không cần train một classifier riêng biệt.

#### Cơ Chế Hoạt Động

Mô hình được train với hai chế độ:
1. **Conditional mode**: Với xác suất \(1 - p_{drop}\), model nhận class label \(y\)
2. **Unconditional mode**: Với xác suất \(p_{drop}\), class label bị drop và thay bằng null embedding

Trong quá trình training:
\[
L = \mathbb{E}_{t, x_0, \epsilon, y} \left[ ||\epsilon - \epsilon_\theta(x_t, t, y)||^2 \right]
\]

Với \(p_{drop}\) (thường là 0.5), model học cả conditional và unconditional generation.

#### Classifier-Free Guidance tại Inference

Tại thời điểm sampling, CFG sử dụng công thức:

\[
\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))
\]

Trong đó:
- \(\epsilon_\theta(x_t, t, \emptyset)\) là prediction không điều kiện
- \(\epsilon_\theta(x_t, t, y)\) là prediction có điều kiện
- \(s\) là guidance scale (thường từ 3-8)

Guidance scale \(s > 1\) làm tăng ảnh hưởng của condition, tạo ra samples phù hợp hơn với class label nhưng có thể giảm diversity.

### 2.2 Kiến Trúc Mô Hình

#### UNet Architecture với Class Embeddings

Mô hình sử dụng `Unet1D_cls_free` với các thành phần chính:

1. **Class Embeddings**:
   ```python
   self.classes_emb = nn.Embedding(num_classes, dim)
   self.null_classes_emb = nn.Parameter(torch.randn(dim))
   ```

2. **Conditional Dropout**:
   ```python
   if cond_drop_prob > 0:
       keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device)
       classes_emb = torch.where(
           rearrange(keep_mask, 'b -> b 1'),
           classes_emb,  # Conditional
           null_classes_emb  # Unconditional
       )
   ```

3. **Class MLP**:
   ```python
   self.classes_mlp = nn.Sequential(
       nn.Linear(dim, classes_dim),
       nn.GELU(),
       nn.Linear(classes_dim, classes_dim)
   )
   ```

#### Architecture Parameters (theo code)

- **Base dimensions**: 64
- **Channel multipliers**: (1, 2, 4, 8)
- **Number of classes**: 5 (Simulated và MITBIH), 9 (UNIMIB)
- **Residual blocks groups**: 8
- **Attention heads**: 4
- **Conditional dropout probability**: 0.5

#### Training Parameters

- **Optimizer**: Adam
- **Batch size**: 64 (theo `ddpm1d_cls_free.py`)
- **Learning rate**: 1e-4
- **Epochs**: 300
- **Diffusion timesteps**: 1000
- **Noise schedule**: cosine
- **Loss**: L1 hoặc L2

### 2.3 Implementation trong Code

#### Training Loop (`ddpm1d_cls_free.py`)

```python
# Model initialization
model = Unet1D_cls_free(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_classes=num_classes,
    cond_drop_prob=0.5,  # 50% unconditional training
    channels=1
)

# Training
loss = diffusion(signals, classes=labels)
```

#### Sampling với CFG

```python
# Conditional generation với guidance scale
sampled_signals = diffusion.sample(
    classes=labels,
    cond_scale=3.0  # Guidance scale
)

# Unconditional generation (có thể dùng với cond_scale=1.0)
unconditional_signals = diffusion.sample(
    classes=dummy_classes,
    cond_scale=1.0
)
```

### 2.4 Forward với Conditional Scale

Cơ chế CFG được implement trong `forward_with_cond_scale`:

```python
def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
    logits = self.forward(*args, **kwargs)  # Conditional prediction
    
    if cond_scale == 1:
        return logits
    
    null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)  # Unconditional
    return null_logits + (logits - null_logits) * cond_scale
```

---

## 3. Signal Conditional Diffusion Models

### 3.1 Lý Thuyết Signal Conditioning

Mô hình Signal Conditional sử dụng một tín hiệu khác làm điều kiện để generate tín hiệu mới. Đây là một dạng của **conditional generation** nhưng điều kiện là một tín hiệu cùng chiều với output, không phải là label rời rạc.

#### Ứng Dụng Chính

1. **Signal Denoising**: Loại bỏ nhiễu từ tín hiệu bị nhiễu
2. **Signal Imputation**: Điền các giá trị bị thiếu trong tín hiệu
3. **Signal Super-resolution**: Tăng độ phân giải của tín hiệu
4. **Individual Signal Generation**: Generate nhiều samples từ một tín hiệu của một subject

#### Forward Process với Condition

Quá trình forward diffusion vẫn giống unconditional, nhưng reverse process có thêm condition:

\[
p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t, c))
\]

Trong đó \(c\) là conditional signal (ví dụ: noisy signal, masked signal, downsampled signal).

#### Training Objective

\[
L = \mathbb{E}_{t, x_0, \epsilon, c} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]
\]

### 3.2 Kiến Trúc Mô Hình

#### UNet với Signal Conditioning

Mô hình sử dụng `Unet1D` với self-conditioning và signal conditioning:

1. **Self-Conditioning**: 
   - 50% thời gian training, model sử dụng prediction từ chính nó làm condition
   - Giúp cải thiện chất lượng generation

2. **Signal Conditioning**:
   - Conditional signal được concatenate hoặc được đưa vào qua cross-attention
   - Trong implementation hiện tại, conditional signal được đưa vào như `x_self_cond`

#### Architecture Parameters

- **Base channels**: 64
- **Channel multipliers**: (1, 2, 4, 8, 8) cho một số tasks
- **Channel multipliers**: (1, 2, 4, 8) cho UNIMIB và MITBIH
- **Residual blocks groups**: 2 (cho signal conditional)
- **Attention heads**: 4
- **Self-conditioning**: True

#### Training Parameters

- **Optimizer**: Adam
- **Batch size**: 32
- **Learning rate**: 1e-4 (theo paper), 3e-4 (theo code)
- **Iterations**: 1,000,000 (theo paper)
- **Epochs**: 100 (theo code)
- **Diffusion timesteps**: 2000 (theo paper), 1000 (theo code)
- **Noise schedule**: linear (theo paper), cosine (theo code)
- **Objective**: pred_v (v-parameterization)
- **Loss**: L1

### 3.3 Implementation trong Code

#### Training Dataset (`ddpm1d_sign_cond.py`)

```python
class TrainingDataset(data.Dataset):
    def __getitem__(self, idx):
        data_dict = {
            'org_data': self.cond_ECG[idx]['org_data'],      # Original signal
            'cond_data': self.cond_ECG[idx]['cond_data']     # Conditional signal
        }
        return data_dict
```

#### Training Loop

```python
# Model với self-conditioning
model = Unet1D(
    dim=64,
    self_condition=True,
    dim_mults=(1, 2, 4, 8),
    channels=1
)

# Diffusion với v-parameterization
diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_length,
    timesteps=1000,
    objective='pred_v'  # V-parameterization
)

# Training
sig1 = data_dict['org_data']  # Target signal
sig2 = data_dict['cond_data']  # Conditional signal
loss = diffusion(sig1, sig2)
```

#### Self-Conditioning Mechanism

Trong `p_losses` của `GaussianDiffusion1D`:

```python
x_self_cond = input_cond  # Ban đầu là conditional signal
if self.self_condition and random() < 0.5:
    with torch.no_grad():
        x_self_cond = self.model_predictions(x, t).pred_x_start
        x_self_cond.detach_()

model_out = self.model(x, t, x_self_cond)
```

#### Sampling

```python
# Sampling với conditional signal
sampled_signals = diffusion.sample(
    batch_size=sample_size,
    input_cond=cond_data  # Conditional signal (noisy, masked, etc.)
)
```

### 3.4 Các Ứng Dụng Cụ Thể

#### 3.4.1 Signal Denoising

- **Input**: Tín hiệu bị nhiễu (thermal noise, electrode noise, motion artifacts)
- **Output**: Tín hiệu đã được làm sạch
- **Condition**: Noisy signal
- **Target**: Clean signal

#### 3.4.2 Signal Imputation

- **Input**: Tín hiệu có missing values (set về 0)
- **Output**: Tín hiệu đã được điền đầy đủ
- **Condition**: Masked signal
- **Target**: Complete signal

#### 3.4.3 Signal Super-resolution

- **Input**: Tín hiệu độ phân giải thấp (downsampled)
- **Output**: Tín hiệu độ phân giải cao
- **Condition**: Low-resolution signal
- **Target**: High-resolution signal

#### 3.4.4 Individual Signal Generation

- **Input**: Một vài tín hiệu từ một subject
- **Output**: Nhiều tín hiệu synthetic với pattern tương tự
- **Condition**: Real signals from individual
- **Target**: Synthetic signals with same characteristics

---

## 4. So Sánh Ba Mô Hình

| Đặc điểm | Unconditional | Label-Conditional | Signal-Conditional |
|----------|---------------|-------------------|-------------------|
| **Condition Type** | Không có | Class labels (discrete) | Signal data (continuous) |
| **Training Data** | Per-class training | All classes với labels | Per-class với signal pairs |
| **Architecture** | UNet đơn giản | UNet + Class embeddings | UNet + Self-conditioning |
| **Conditional Dropout** | Không áp dụng | 0.5 (CFG) | Không áp dụng |
| **Guidance Scale** | N/A | 3-8 (CFG) | N/A |
| **Objective** | pred_noise | pred_noise | pred_v |
| **Timesteps** | 1000 | 1000 | 2000 (paper) / 1000 (code) |
| **Noise Schedule** | Cosine | Cosine | Linear (paper) / Cosine (code) |
| **Use Cases** | General generation | Class-specific generation | Denoising, imputation, super-resolution |

---

## 5. Loss Functions và Objectives

### 5.1 Prediction Objectives

Ba loại objectives được sử dụng:

1. **pred_noise** (ε-prediction):
   - Dự đoán nhiễu \(\epsilon\) đã được thêm vào
   - Phổ biến nhất, ổn định

2. **pred_x0** (x₀-prediction):
   - Dự đoán trực tiếp tín hiệu gốc \(x_0\)
   - Có thể không ổn định ở timesteps cao

3. **pred_v** (v-parameterization):
   - Dự đoán \(v = \alpha_t \epsilon - \sqrt{1-\alpha_t} x_0\)
   - Được sử dụng trong Signal Conditional model
   - Cân bằng giữa noise và signal prediction

### 5.2 Loss Types

- **L1 Loss**: Robust với outliers, được sử dụng trong hầu hết các models
- **L2 Loss**: Smooth hơn, nhạy cảm với outliers

---

## 6. Noise Schedules

### 6.1 Cosine Schedule

Được sử dụng trong Unconditional và Label-Conditional models:

\[
\beta_t = \text{clip}(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0, 0.999)
\]

Với:
\[
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
\]

Cosine schedule thêm nhiễu chậm hơn ở đầu và nhanh hơn ở cuối, phù hợp với biomedical signals.

### 6.2 Linear Schedule

Được sử dụng trong Signal Conditional model (theo paper):

\[
\beta_t = \text{linear\_interpolate}(\beta_{start}, \beta_{end}, t/T)
\]

Linear schedule đơn giản hơn, thêm nhiễu đều đặn.

---

## 7. Sampling Methods

### 7.1 DDPM Sampling (Standard)

Sampling đầy đủ qua tất cả timesteps:

```python
for t in reversed(range(0, num_timesteps)):
    img, x_start = p_sample(img, t, condition)
```

### 7.2 DDIM Sampling (Deterministic)

Sampling nhanh hơn với ít timesteps hơn:

```python
sampling_timesteps < timesteps  # Ví dụ: 50 steps thay vì 1000
```

DDIM cho phép deterministic sampling và nhanh hơn nhiều.

---

## 8. Kết Luận

Ba mô hình trong BioDiffusion framework cung cấp các khả năng generation linh hoạt:

1. **Unconditional Model**: Phù hợp cho general data augmentation và exploration
2. **Label-Conditional Model**: Phù hợp cho class-specific generation với control tốt
3. **Signal Conditional Model**: Phù hợp cho các tasks như denoising, imputation, và super-resolution

Mỗi mô hình có architecture và training strategy riêng, được tối ưu cho các use cases cụ thể trong biomedical signal processing.

---

## References

- Paper: "BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis" (arXiv:2401.10282v2)
- Code Implementation: `src/signal/ddpm1d_cls_free.py`, `src/signal/ddpm1d_sign_cond.py`
- Modules: `src/signal/modules/modules1D_cls_free.py`, `src/signal/modules/modules1D_cond.py`

---

## Prompt cho AI Agents để Generate Academic Report

```
Bạn là một chuyên gia về diffusion models và biomedical signal processing. Nhiệm vụ của bạn là viết một báo cáo học thuật (academic report) về lý thuyết và kiến trúc của ba mô hình diffusion trong framework BioDiffusion.

Dựa trên markdown document đã được cung cấp, hãy viết một báo cáo học thuật với các yêu cầu sau:

1. **Cấu trúc báo cáo**:
   - Abstract/Summary
   - Introduction và Background
   - Methodology chi tiết cho từng mô hình
   - Mathematical formulations đầy đủ
   - Architecture descriptions
   - Training procedures
   - Experimental setup (nếu có)
   - Discussion và Analysis
   - Conclusion và Future Work

2. **Yêu cầu về phong cách học thuật**:
   - Sử dụng ngôn ngữ formal, chuyên nghiệp
   - Trích dẫn các công thức toán học với ký hiệu chuẩn
   - Giải thích rõ ràng các khái niệm kỹ thuật
   - So sánh và phân tích sâu các phương pháp
   - Sử dụng các thuật ngữ chuyên ngành chính xác

3. **Nội dung cần bao gồm**:
   - Lý thuyết diffusion models cơ bản
   - Chi tiết về forward và reverse diffusion processes
   - Classifier-Free Guidance mechanism
   - Signal conditioning strategies
   - Architecture details (UNet variants)
   - Training objectives và loss functions
   - Noise schedules và sampling methods
   - So sánh giữa ba mô hình

4. **Format yêu cầu**:
   - Sử dụng LaTeX notation cho công thức toán học
   - Có các sections và subsections rõ ràng
   - Include code snippets nếu cần thiết (với giải thích)
   - Có bảng so sánh và diagrams nếu có thể

5. **Độ sâu phân tích**:
   - Giải thích tại sao mỗi design choice được thực hiện
   - Phân tích trade-offs giữa các phương pháp
   - Thảo luận về limitations và advantages
   - Đề xuất improvements hoặc extensions

Hãy viết báo cáo một cách toàn diện, chính xác về mặt kỹ thuật, và phù hợp với tiêu chuẩn của một academic paper hoặc technical report.
```

