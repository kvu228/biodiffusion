# Kiến trúc UNet trong BioDiffusion: Tài liệu Chi tiết

## Tổng quan

BioDiffusion sử dụng kiến trúc U-Net làm backbone cho các mô hình diffusion, được tối ưu hóa đặc biệt cho cả tín hiệu 1D (biomedical signals) và hình ảnh 2D. Tài liệu này mô tả chi tiết các biến thể của UNet, các thành phần cấu thành, và cách chúng được sử dụng trong các tác vụ generation khác nhau.

## 1. Kiến trúc UNet 1D cho Biomedical Signals

### 1.1 Tổng quan

UNet 1D được thiết kế để xử lý các tín hiệu y sinh học như ECG, EEG, và accelerometer signals. Kiến trúc này sử dụng các phép toán convolution 1D thay vì 2D, phù hợp với bản chất tuần tự của tín hiệu thời gian.

### 1.2 Cấu trúc tổng thể

```
Input Signal (B, C, L)
    ↓
[Initial Conv1d] → (B, dim, L)
    ↓
[Encoder Path - Downsampling]
    ├── ResnetBlock + Attention
    ├── ResnetBlock + Attention  
    ├── Downsample (L/2)
    ├── ResnetBlock + Attention
    ├── ResnetBlock + Attention
    ├── Downsample (L/4)
    └── ... (tiếp tục theo dim_mults)
    ↓
[Bottleneck]
    ├── ResnetBlock + Attention
    └── ResnetBlock
    ↓
[Decoder Path - Upsampling]
    ├── Concatenate với skip connection
    ├── ResnetBlock + Attention
    ├── Concatenate với skip connection
    ├── ResnetBlock + Attention
    ├── Upsample (L*2)
    └── ... (mirror encoder)
    ↓
[Final ResBlock + Conv1d] → (B, C, L)
```

### 1.3 Các thành phần chính

#### 1.3.1 Initial Convolution

```python
self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
```

- **Mục đích**: Project input signal vào không gian feature ban đầu
- **Kernel size**: 7 với padding=3 để giữ nguyên chiều dài
- **Input channels**: Có thể là 1 (single channel) hoặc nhiều hơn (multivariate signals)
- **Output**: `(B, init_dim, L)` với `init_dim` thường là 64

#### 1.3.2 Encoder (Downsampling Path)

Encoder path giảm resolution của tín hiệu và tăng số channels:

```python
for ind, (dim_in, dim_out) in enumerate(in_out):
    is_last = ind >= (num_resolutions - 1)
    
    self.downs.append(nn.ModuleList([
        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
    ]))
```

**Đặc điểm**:
- Mỗi resolution level có 2 ResnetBlocks + 1 Attention layer
- Downsampling giảm chiều dài tín hiệu xuống 1/2
- Channels tăng theo `dim_mults`: `(1, 2, 4, 8)` → `[64, 128, 256, 512]`
- Skip connections được lưu lại cho decoder

**Ví dụ với `dim_mults=(1, 2, 4, 8)` và `seq_length=128`**:
- Level 0: `(B, 64, 128)` → `(B, 64, 128)` (no downsampling)
- Level 1: `(B, 64, 128)` → `(B, 128, 64)` (downsample)
- Level 2: `(B, 128, 64)` → `(B, 256, 32)` (downsample)
- Level 3: `(B, 256, 32)` → `(B, 512, 16)` (downsample)

#### 1.3.3 Bottleneck

Bottleneck xử lý features ở resolution thấp nhất:

```python
mid_dim = dims[-1]  # 512
self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
```

- **Mục đích**: Capture global context ở resolution thấp nhất
- **Attention**: Sử dụng full attention (không phải linear attention) để học long-range dependencies
- **Output**: `(B, 512, 16)` (với seq_length=128)

#### 1.3.4 Decoder (Upsampling Path)

Decoder path tăng resolution và giảm channels, kết hợp với skip connections:

```python
for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
    is_last = ind == (len(in_out) - 1)
    
    self.ups.append(nn.ModuleList([
        ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
        ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
        Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
    ]))
```

**Đặc điểm**:
- **Skip connections**: Concatenate với features từ encoder ở cùng resolution
- **Channel handling**: Input là `dim_out + dim_in` (từ decoder + skip connection)
- **Upsampling**: Tăng chiều dài lên 2x bằng nearest neighbor interpolation

**Ví dụ**:
- Level 3: `(B, 512, 16)` + skip `(B, 256, 32)` → `(B, 256, 32)` → upsample → `(B, 256, 64)`
- Level 2: `(B, 256, 64)` + skip `(B, 128, 64)` → `(B, 128, 64)` → upsample → `(B, 128, 128)`
- Level 1: `(B, 128, 128)` + skip `(B, 64, 128)` → `(B, 64, 128)`

#### 1.3.5 Final Layers

```python
self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
```

- **Final ResBlock**: Kết hợp với initial feature `r` (skip connection từ đầu)
- **Final Conv**: Project về output channels (thường bằng input channels)

### 1.4 Time Embedding

Time embedding được inject vào mỗi ResnetBlock để model biết timestep hiện tại trong quá trình diffusion:

```python
time_dim = dim * 4  # 256 với dim=64

self.time_mlp = nn.Sequential(
    SinusoidalPosEmb(dim),           # Sinusoidal encoding
    nn.Linear(fourier_dim, time_dim), # Project to time_dim
    nn.GELU(),
    nn.Linear(time_dim, time_dim)
)
```

**Sinusoidal Positional Encoding**:
```python
def forward(self, x):
    device = x.device
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb
```

**Công thức**: Tương tự transformer positional encoding:
- `PE(pos, 2i) = sin(pos / 10000^(2i/d))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`

**Injection vào ResnetBlock**:
```python
# Trong ResnetBlock.forward()
if exists(self.mlp) and exists(time_emb):
    time_emb = self.mlp(time_emb)  # (B, dim_out * 2)
    time_emb = rearrange(time_emb, 'b c -> b c 1')
    scale_shift = time_emb.chunk(2, dim=1)  # Split thành scale và shift
    
    # Apply trong Block
    x = x * (scale + 1) + shift  # Feature-wise affine transformation
```

### 1.5 Attention Mechanisms

#### 1.5.1 Linear Attention

Sử dụng trong encoder/decoder để giảm computational cost:

```python
class LinearAttention(nn.Module):
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        
        q = q.softmax(dim=-2)  # Normalize query
        k = k.softmax(dim=-1)  # Normalize key
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        return self.to_out(out)
```

**Đặc điểm**:
- **Complexity**: O(n) thay vì O(n²) của standard attention
- **Normalization**: Softmax trên query và key riêng biệt
- **Efficiency**: Phù hợp cho sequences dài

#### 1.5.2 Full Attention

Sử dụng trong bottleneck:

```python
class Attention(nn.Module):
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        
        q = q * self.scale  # Scale by sqrt(d_k)
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h d j', attn, v)
        return self.to_out(out)
```

**Đặc điểm**:
- **Standard scaled dot-product attention**
- **Complexity**: O(n²) nhưng ở resolution thấp nên vẫn hiệu quả
- **Mục đích**: Capture global dependencies tốt hơn

### 1.6 ResnetBlock

Building block cơ bản của UNet:

```python
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)  # Residual connection
```

**Components**:
- **Block**: Weight-standardized Conv1d + GroupNorm + SiLU
- **Time conditioning**: Feature-wise affine transformation (scale & shift)
- **Residual connection**: Giúp gradient flow tốt hơn

## 2. Biến thể Conditional của UNet 1D

### 2.1 Label-Conditional UNet (Classifier-Free Guidance)

#### 2.1.1 Kiến trúc

```python
class Unet1D_cls_free(nn.Module):
    def __init__(self, dim, num_classes, cond_drop_prob=0.5, ...):
        # Class embeddings
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        
        classes_dim = dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )
```

#### 2.1.2 Classifier-Free Guidance

**Training**:
```python
# 50% label dropout
if cond_drop_prob > 0:
    keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
    null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
    
    classes_emb = torch.where(
        rearrange(keep_mask, 'b -> b 1'),
        classes_emb,  # Use real class embedding
        null_classes_emb  # Use null embedding
    )
```

**Inference với CFG**:
```python
def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
    logits = self.forward(*args, **kwargs)  # Conditional prediction
    
    if cond_scale == 1:
        return logits
    
    null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)  # Unconditional
    return null_logits + (logits - null_logits) * cond_scale
```

**Công thức CFG**:
```
pred = uncond + scale * (cond - uncond)
```

- `scale=1`: Chỉ conditional
- `scale>1`: Amplify class signal (thường dùng 3-8)

#### 2.1.3 Class Embedding Injection

Class embedding được inject vào ResnetBlock cùng với time embedding:

```python
# Trong ResnetBlock
cond_emb = torch.cat([time_emb, class_emb], dim=-1)  # Concatenate
cond_emb = self.mlp(cond_emb)  # Project
scale_shift = cond_emb.chunk(2, dim=1)  # Split
```

### 2.2 Signal-Conditional UNet

#### 2.2.1 Kiến trúc

Sử dụng self-conditioning với conditional input:

```python
class Unet1D(nn.Module):
    def __init__(self, ..., self_condition=False):
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
```

#### 2.2.2 Forward Pass

```python
def forward(self, x, time, x_self_cond=None):
    if self.self_condition:
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim=1)  # Concatenate conditional signal
    
    x = self.init_conv(x)
    # ... rest of UNet
```

**Ứng dụng**:
- **Signal denoising**: Conditional input là noisy signal
- **Signal imputation**: Conditional input là signal với missing values
- **Super-resolution**: Conditional input là downsampled signal

## 3. Kiến trúc UNet 2D cho Images

### 3.1 Tổng quan

UNet 2D được sử dụng trong `src/modules/modules.py` cho image generation tasks. Kiến trúc tương tự UNet 1D nhưng sử dụng Conv2d và xử lý spatial dimensions.

### 3.2 Cấu trúc

```python
class UNet(nn.Module):
    def __init__(self, c_in, c_out, time_dim=256):
        # Input layer
        self.inc = DoubleConv(c_in, 64)
        
        # Encoder
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)
        
        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        # Decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        
        # Output
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
```

### 3.3 Các thành phần

#### 3.3.1 DoubleConv

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),  # InstanceNorm
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
```

#### 3.3.2 Down Block

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb  # Additive time embedding
```

#### 3.3.3 Up Block

```python
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)  # Skip connection
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
```

### 3.4 Conditional UNet 2D

```python
class UNet_conditional(UNet):
    def __init__(self, c_in, c_out, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
    
    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        if y is not None:
            t += self.label_emb(y)  # Additive label conditioning
        
        return self.unet_forwad(x, t)
```

## 4. So sánh UNet 1D và 2D

| Đặc điểm | UNet 1D | UNet 2D |
|----------|---------|---------|
| **Input shape** | `(B, C, L)` | `(B, C, H, W)` |
| **Convolution** | Conv1d | Conv2d |
| **Downsampling** | Conv1d stride=2 | MaxPool2d + Conv2d |
| **Upsampling** | Nearest + Conv1d | Bilinear + Conv2d |
| **Attention** | Linear + Full | SelfAttention (Multihead) |
| **Time embedding** | Feature-wise affine | Additive |
| **Normalization** | GroupNorm | GroupNorm/InstanceNorm |
| **Ứng dụng** | Biomedical signals | Images |

## 5. Hyperparameters và Cấu hình

### 5.1 Cấu hình cho 1D Signals

**Theo paper BioDiffusion**:

**Unconditional Model**:
- Base channels: 64
- Channel multipliers: `(1, 2, 4, 8, 8)` (Simulated) hoặc `(1, 2, 4, 8)` (UNIMIB, MITBIH)
- Residual blocks groups: 8
- Attention heads: 4
- Sequence length: 128 (UNIMIB), 144 (MITBIH), 512 (Simulated)

**Label-Conditional Model**:
- Base dimensions: 64
- Channel multipliers: Tương tự unconditional
- Number classes: 5 (Simulated, MITBIH), 9 (UNIMIB)
- Condition dropout probability: 0.5

**Signal-Conditional Model**:
- Base channels: 64
- Channel multipliers: `(1, 2, 4, 8, 8)`
- Residual blocks groups: 2
- Attention heads: 4

### 5.2 Cấu hình cho 2D Images

**CIFAR-10**:
- Input/Output channels: 3 (RGB)
- Base channels: 64
- Channel progression: 64 → 128 → 256 → 256
- Time dimension: 256
- Image size: 32×32

## 6. Training và Optimization

### 6.1 Loss Functions

**1D Signals**:
- L1 Loss: `nn.L1Loss()` (default)
- L2 Loss: `nn.MSELoss()` (optional)
- P2 Loss Weight: Optional reweighting based on timestep

**2D Images**:
- L1 Loss: Standard cho DDPM
- L2 Loss: Alternative

### 6.2 Optimizers

- **Optimizer**: Adam hoặc AdamW
- **Learning rate**: 
  - 1D: 1e-4 (signal-conditional), 3e-4 (label-conditional)
  - 2D: 3e-4
- **Batch size**: 32-64

### 6.3 Diffusion Parameters

**Timesteps**:
- Standard: 1000
- Signal-conditional: 2000

**Noise Schedule**:
- Cosine: Default cho most cases
- Linear: Alternative

## 7. Ứng dụng trong BioDiffusion

### 7.1 Unconditional Generation

Tạo tín hiệu mới không có điều kiện:

```python
model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
diffusion = GaussianDiffusion1D(model, seq_length=128, timesteps=1000)
sampled_signals = diffusion.sample(batch_size=10)
```

### 7.2 Label-Conditional Generation

Tạo tín hiệu theo class label:

```python
model = Unet1D_cls_free(
    dim=64, 
    num_classes=5, 
    cond_drop_prob=0.5,
    channels=1
)
diffusion = GaussianDiffusion1D_cls_free(model, seq_length=128, timesteps=1000)
labels = torch.tensor([0, 1, 2, 3, 4])
sampled_signals = diffusion.sample(classes=labels, cond_scale=3.0)
```

### 7.3 Signal-Conditional Generation

**Denoising**:
```python
model = Unet1D(dim=64, self_condition=True, channels=1)
diffusion = GaussianDiffusion1D(model, seq_length=128, timesteps=2000)
denoised = diffusion.sample(input_cond=noisy_signal)
```

**Imputation**:
```python
# Conditional input là signal với missing values
imputed = diffusion.sample(input_cond=signal_with_blanks)
```

**Super-resolution**:
```python
# Conditional input là downsampled signal
high_res = diffusion.sample(input_cond=low_res_signal)
```

## 8. Best Practices và Tips

### 8.1 Architecture Design

1. **Channel multipliers**: Bắt đầu với `(1, 2, 4, 8)` cho most cases
2. **Attention placement**: Đặt ở resolution thấp để hiệu quả
3. **Residual connections**: Luôn sử dụng trong ResnetBlocks
4. **Skip connections**: Critical cho reconstruction quality

### 8.2 Training Tips

1. **Time embedding**: Sinusoidal encoding ổn định hơn learned
2. **Normalization**: GroupNorm tốt cho batch size nhỏ
3. **Learning rate**: Warm-up có thể giúp training ổn định
4. **Gradient clipping**: Có thể cần thiết cho sequences dài

### 8.3 Conditional Generation

1. **CFG scale**: 3-8 thường cho kết quả tốt
2. **Condition dropout**: 0.5 là optimal cho classifier-free
3. **Label embedding**: Cùng dimension với time embedding để dễ fusion

## 9. Tài liệu tham khảo

- **Paper**: BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis (arXiv:2401.10282v2)
- **Code repository**: `src/signal/Unet1D.py`, `src/signal/modules/modules1D_cls_free.py`
- **2D Implementation**: `src/modules/modules.py`

## 10. Prompt cho AI để Generate Báo cáo Học thuật

---

## PROMPT CHO AI: GENERATE BÁO CÁO HỌC THUẬT VỀ KIẾN TRÚC UNET TRONG BIODIFFUSION

Bạn là một chuyên gia về deep learning và diffusion models. Hãy viết một báo cáo học thuật chi tiết về kiến trúc UNet được sử dụng trong mô hình BioDiffusion dựa trên thông tin sau:

### Yêu cầu:

1. **Văn phong học thuật**: Sử dụng ngôn ngữ chuyên ngành, cấu trúc rõ ràng, và tuân thủ format của paper khoa học.

2. **Nội dung cần bao gồm**:

   **Phần 1: Architecture Design**
   - Mô tả chi tiết kiến trúc UNet 1D cho biomedical signals
   - Encoder-decoder structure với skip connections
   - Bottleneck design và attention mechanisms
   - So sánh với UNet 2D cho images

   **Phần 2: Key Components**
   - ResnetBlocks với time embedding injection
   - Attention mechanisms (Linear Attention và Full Attention)
   - Downsampling và upsampling strategies
   - Time embedding và positional encoding

   **Phần 3: Conditional Variants**
   - Label-conditional UNet với classifier-free guidance
   - Signal-conditional UNet cho denoising, imputation, super-resolution
   - Cơ chế conditioning và fusion strategies

   **Phần 4: Technical Details**
   - Hyperparameters và cấu hình cho các datasets khác nhau
   - Training procedures và optimization strategies
   - Loss functions và noise schedules

   **Phần 6: Applications**
   - Unconditional generation
   - Label-conditional generation
   - Signal-conditional tasks (denoising, imputation, super-resolution)

3. **Thông tin kỹ thuật cần tham khảo**:
   - File code: `src/signal/Unet1D.py`, `src/signal/modules/modules1D_cls_free.py`, `src/modules/modules.py`
   - Paper: BioDiffusion (arXiv:2401.10282v2)
   - Architecture details từ tài liệu này

4. **Format**:
   - Sử dụng LaTeX notation cho công thức toán học
   - Include figures/diagrams mô tả architecture (mô tả bằng text nếu không có hình)
   - Citations và references đúng format
   - Section numbering rõ ràng

5. **Độ dài**: Khoảng 10-15 trang (không tính references)

6. **Ngôn ngữ**: tiếng việt

Hãy viết báo cáo với độ chính xác cao về technical details, sử dụng terminology chuyên ngành đúng chuẩn, và trình bày một cách logic, dễ hiểu cho người đọc có background về deep learning.

---

