"""
Standalone Lada BasicVSR++ model adapter for nvRestorer.

This module provides a minimal, standalone version of Lada's BasicVSR++ 
implementation without requiring full mmagic/mmengine dependencies.

Directory structure:
    nvRestorer/models/lada/
    ├── __init__.py (this file)
    ├── basicvsr_plusplus_net.py (from Lada, with import fixes)
    ├── deformconv.py (from Lada - just a stub, real work done by torchvision)
    ├── flow_warp.py (from Lada)
    └── model_utils.py (from Lada, with import fixes)

Usage:
    from nvRestorer.models.lada import BasicVSRPlusPlusNet
    
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        spynet_pretrained=None
    )
    
    # Load Lada weights
    state_dict = torch.load("lada_weights.pth", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Strip generator_ema prefix
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("generator_ema."):
            clean_state[k[14:]] = v  # Remove "generator_ema."
        else:
            clean_state[k] = v
    
    model.load_state_dict(clean_state, strict=True)
    model.eval()
"""

import torch
import torch.nn as nn

# ============================================================================
# MMEngine Dependency Stubs
# ============================================================================
# These stubs replace mmengine dependencies with minimal implementations

class BaseModule(nn.Module):
    """
    Minimal BaseModule stub to replace mmengine.model.BaseModule.
    
    The original BaseModule provides initialization configuration management,
    but we handle weight initialization manually in each module.
    """
    def __init__(self, init_cfg=None):
        super().__init__()
        # Ignore init_cfg - weight initialization is handled in each module's init_weights()

class MMLogger:
    """
    Minimal MMLogger stub to replace mmengine.MMLogger.
    
    Provides basic logging functionality using Python's standard logging module.
    """
    @staticmethod
    def get_current_instance():
        import logging
        return logging.getLogger(__name__)

def load_checkpoint(model, filename, map_location='cpu', strict=True, logger=None, **kwargs):
    """
    Minimal load_checkpoint stub to replace mmengine.runner.load_checkpoint.
    
    Loads a PyTorch checkpoint file and applies it to the model.
    
    Args:
        model: PyTorch model
        filename: Path to checkpoint file
        map_location: Device to map tensors to
        strict: Whether to strictly enforce state_dict keys match
        logger: Optional logger instance
        
    Returns:
        Missing and unexpected keys from load_state_dict
    """
    if logger:
        logger.info(f"Loading checkpoint from {filename}")
    
    try:
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filename, map_location=map_location)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    return model.load_state_dict(state_dict, strict=strict)

class Registry:
    """
    Minimal Registry stub to replace mmengine.registry.Registry.
    
    Provides module registration functionality used by MMagic's decorator pattern.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = {}
    
    def register_module(self, name=None, force=False, module=None):
        """
        Decorator to register a module.
        
        Usage:
            @MODELS.register_module()
            class MyModel(nn.Module):
                pass
        """
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            if not force and module_name in self._module_dict:
                raise KeyError(f'{module_name} is already registered in {self._name}')
            self._module_dict[module_name] = cls
            return cls
        
        if module is not None:
            return _register(module)
        return _register
    
    def build(self, cfg, *args, **kwargs):
        """Build a module from config dict."""
        if isinstance(cfg, dict):
            cfg = cfg.copy()
            module_type = cfg.pop('type')
            if module_type not in self._module_dict:
                raise KeyError(f'{module_type} is not registered in {self._name}')
            return self._module_dict[module_type](*args, **cfg, **kwargs)
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

# Create global MODELS registry (used by @MODELS.register_module() decorator)
MODELS = Registry('models')

# ============================================================================
# Weight Initialization Function Stubs
# ============================================================================
# These replace mmengine.model.weight_init functions

def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='normal'):
    """
    Kaiming initialization for convolutional and linear layers.
    
    Args:
        module: nn.Module to initialize
        a: Negative slope for leaky_relu/relu
        mode: 'fan_in' or 'fan_out'
        nonlinearity: Activation function name
        bias: Bias initialization value
        distribution: 'normal' or 'uniform'
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val=0, bias=0):
    """
    Constant initialization for weights and biases.
    
    Args:
        module: nn.Module to initialize (or just a parameter)
        val: Value for weight initialization
        bias: Value for bias initialization
    """
    if isinstance(module, nn.Module):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, nn.Parameter):
        nn.init.constant_(module, val)

# BatchNorm stub (used by model_utils.py)
class _BatchNorm(nn.modules.batchnorm._BatchNorm):
    """Stub for mmengine.utils.dl_utils.parrots_wrapper._BatchNorm"""
    pass

# ============================================================================
# Import Local Modules
# ============================================================================

from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
from .flow_warp import flow_warp
from .model_utils import default_init_weights, make_layer
from .deformconv import ModulatedDeformConv2d

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Model
    'BasicVSRPlusPlusNet',
    
    # Utilities
    'flow_warp',
    'default_init_weights',
    'make_layer',
    'ModulatedDeformConv2d',
    
    # MMEngine stubs (needed by imported modules)
    'BaseModule',
    'MMLogger',
    'load_checkpoint',
    'MODELS',
    'kaiming_init',
    'constant_init',
    '_BatchNorm',
]
