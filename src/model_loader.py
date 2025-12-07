"""Utility class for loading pre-trained models"""

import torch
import os
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Note: UNet and UNet_conditional from modules.modules are deprecated
# Only 1D models (modules1D_cls_free, modules1D_cond) are supported
UNet = None
UNet_conditional = None

try:
    from modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
    # Map old module path for checkpoint loading compatibility
    # Checkpoints saved with 'modules.modules1D_cls_free' can now be loaded
    sys.modules['modules.modules1D_cls_free'] = sys.modules['modules.modules1D_cls_free']
except ImportError:
    try:
        from src.modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
        # Map old module path for checkpoint loading compatibility
        sys.modules['modules.modules1D_cls_free'] = sys.modules['src.modules.modules1D_cls_free']
    except ImportError:
        Unet1D_cls_free = None
        GaussianDiffusion1D_cls_free = None

try:
    from modules.modules1D_cond import Unet1D, GaussianDiffusion1D
except ImportError:
    try:
        from src.modules.modules1D_cond import Unet1D, GaussianDiffusion1D
    except ImportError:
        Unet1D = None
        GaussianDiffusion1D = None


class ModelLoader:
    """Utility class for loading pre-trained models"""
    
    @staticmethod
    def load_unconditional_model(checkpoint_path, device="cuda"):
        """
        Load unconditional UNet model
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded model in eval mode
            
        Note: This method is deprecated. Use 1D models instead.
        """
        raise NotImplementedError("UNet (2D) models are deprecated. Please use 1D models (Unet1D_cls_free or Unet1D) instead.")
    
    @staticmethod
    def load_conditional_model(checkpoint_path, num_classes=10, device="cuda", use_ema=False):
        """
        Load conditional UNet model with optional EMA
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            num_classes: Number of classes
            device: Device to load model on
            use_ema: Whether to load EMA model
            
        Returns:
            Loaded model(s) in eval mode
            
        Note: This method is deprecated. Use 1D models instead.
        """
        raise NotImplementedError("UNet_conditional (2D) models are deprecated. Please use 1D models (Unet1D_cls_free or Unet1D) instead.")
    
    @staticmethod
    def load_conditional_model_with_ema(checkpoint_path, num_classes=10, device="cuda"):
        """
        Load both regular and EMA conditional models
        
        Args:
            checkpoint_path: Path to checkpoint directory
            num_classes: Number of classes
            device: Device to load model on
            
        Returns:
            Tuple of (model, ema_model) in eval mode
            
        Note: This method is deprecated. Use 1D models instead.
        """
        raise NotImplementedError("UNet_conditional (2D) models are deprecated. Please use 1D models (Unet1D_cls_free or Unet1D) instead.")
    
    @staticmethod
    def list_available_models(models_dir="models"):
        """
        List all available model checkpoints
        
        Args:
            models_dir: Directory containing model checkpoints
            
        Returns:
            Dictionary mapping model names to paths and metadata
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            return {}
        
        models = {}
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Check for checkpoint files
                ckpt_path = model_dir / "ckpt.pt"
                ema_path = model_dir / "ema_ckpt.pt"
                
                if ckpt_path.exists() or ema_path.exists():
                    models[model_dir.name] = {
                        "path": str(model_dir),
                        "has_ema": ema_path.exists(),
                        "has_regular": ckpt_path.exists()
                    }
        
        return models
    
    @staticmethod
    def load_1d_cls_free_model(checkpoint_path, num_classes=5, seq_length=128, device="cuda", 
                                dim=64, dim_mults=(1, 2, 4, 8), cond_drop_prob=0.5, channels=1):
        """
        Load 1D classifier-free guidance model
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            num_classes: Number of classes
            seq_length: Sequence length
            device: Device to load model on
            dim: Model dimension
            dim_mults: Dimension multipliers
            cond_drop_prob: Conditional dropout probability
            channels: Number of channels
            
        Returns:
            Tuple of (model, diffusion) in eval mode
        """
        if Unet1D_cls_free is None:
            raise ImportError("Cannot import Unet1D_cls_free. Make sure signal modules are available.")
        
        model = Unet1D_cls_free(
            dim=dim,
            dim_mults=dim_mults,
            num_classes=num_classes,
            cond_drop_prob=cond_drop_prob,
            channels=channels
        ).to(device)
        
        # Handle both file path and directory path
        if os.path.isdir(checkpoint_path):
            ckpt_path = os.path.join(checkpoint_path, "checkpoint.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_path, "ckpt.pt")
        else:
            ckpt_path = checkpoint_path
        
        # Load checkpoint with weights_only=False to handle old checkpoints
        # The module path mapping is done at the top of the file
        # This allows loading checkpoints saved with 'modules.modules1D_cls_free' path
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                # If checkpoint contains model object, extract its state_dict
                if hasattr(checkpoint['model'], 'state_dict'):
                    state_dict = checkpoint['model'].state_dict()
                else:
                    state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            # If checkpoint is a model object, extract state_dict
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Create diffusion model
        diffusion = GaussianDiffusion1D_cls_free(
            model,
            seq_length=seq_length,
            timesteps=1000
        ).to(device)
        diffusion.eval()
        
        return model, diffusion
    
    @staticmethod
    def load_1d_signal_cond_model(checkpoint_path, seq_length=128, device="cuda", 
                                   dim=64, dim_mults=(1, 2, 4, 8), channels=1, 
                                   self_condition=True, objective='pred_v'):
        """
        Load 1D signal conditional model (for denoising, imputation, super-resolution)
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            seq_length: Sequence length
            device: Device to load model on
            dim: Model dimension
            dim_mults: Dimension multipliers
            channels: Number of channels
            self_condition: Whether to use self-conditioning
            objective: Objective type ('pred_noise', 'pred_x0', 'pred_v')
            
        Returns:
            Tuple of (model, diffusion) in eval mode
        """
        if Unet1D is None or GaussianDiffusion1D is None:
            raise ImportError("Cannot import Unet1D or GaussianDiffusion1D. Make sure signal conditional modules are available.")
        
        model = Unet1D(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
            self_condition=self_condition
        ).to(device)
        
        # Handle both file path and directory path
        if os.path.isdir(checkpoint_path):
            ckpt_path = os.path.join(checkpoint_path, "checkpoint.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_path, "ckpt.pt")
        else:
            ckpt_path = checkpoint_path
        
        # Load checkpoint with weights_only=False to handle old checkpoints
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
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
        else:
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Create diffusion model
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=seq_length,
            timesteps=1000,
            objective=objective
        ).to(device)
        diffusion.eval()
        
        return model, diffusion
    
    @staticmethod
    def list_available_1d_models(checkpoint_dir="src/signal/checkpoint"):
        """
        List all available 1D model checkpoints from signal directory
        
        Args:
            checkpoint_dir: Directory containing 1D model checkpoints
            
        Returns:
            Dictionary mapping model names to paths and metadata
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return {}
        
        models = {}
        for model_dir in checkpoint_dir.iterdir():
            if model_dir.is_dir():
                # Check for checkpoint files
                ckpt_path = model_dir / "checkpoint.pt"
                if not ckpt_path.exists():
                    ckpt_path = model_dir / "ckpt.pt"
                
                if ckpt_path.exists():
                    models[model_dir.name] = {
                        "path": str(model_dir),
                        "checkpoint_path": str(ckpt_path)
                    }
        
        return models

