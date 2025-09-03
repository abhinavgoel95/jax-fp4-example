"""
FP4 quantization kernels for JAX.
Provides JAX wrappers for PyTorch-based FP4 quantization kernels using DLPack for interoperability.
"""

import jax
import jax.numpy as jnp
from typing import Optional
import transformer_engine as te

from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.quantize import (
    ScaledTensor,
    ScaledTensor1x,
    ScaledTensor2x,
    GroupedScaledTensor1x,
    ScalingMode,
    QuantizerFactory,
    QuantizeLayout,
)


def quantize_e2m1_scaled(x):
    """
    Quantize input array using E2M1 kernel.
    
    Args:
        x: Input JAX array
        amax: Absolute maximum value (computed if None)
        use_rs: Whether to use stochastic rounding (default: False)
    
    Returns:
        Quantized JAX array
    """
    te_quantizer = QuantizerFactory.create(n_quantizers=1, q_dtype=jnp.float4_e2m1fn, scaling_mode=ScalingMode.NVFP4_1D_SCALING, q_layout=QuantizeLayout.ROWWISE)
    q = tex.quantize(x, quantizer=te_quantizer)
    return q.data, q.scale_inv, q.amax


def calculate_amax(a):
    """Calculate absolute maximum value for quantization scaling."""
    amax_a = jnp.max(jnp.abs(a)).astype(jnp.float32)
    return amax_a


def pad_tensor(
    tensor: jnp.ndarray, 
    row_divisor: int = 32, 
    col_divisor: Optional[int] = None
) -> jnp.ndarray:
    """
    JAX equivalent of _pad_tensor from PyTorch quantization code.
    
    Pad the tensor because of the PSX clippy requirements.
    Ensures the number of rows is a multiple of row_divisor and
    optionally ensures columns are a multiple of col_divisor.
    Returns the original tensor if padding is not needed.
    
    Assumes tensor is always 2D.
    
    Args:
        tensor: 2D JAX array to pad
        row_divisor: Rows must be multiple of this value
        col_divisor: Columns must be multiple of this value (optional)
        
    Returns:
        Padded tensor
    """
    assert tensor.ndim == 2, "only supports 2D tensors"
    M, N = tensor.shape
    
    # Calculate row padding
    padding_needed_rows = 0
    if M % row_divisor != 0:
        padding_needed_rows = row_divisor - (M % row_divisor)
    
    # Calculate column padding if col_divisor is provided
    padding_needed_cols = 0
    if col_divisor is not None and N % col_divisor != 0:
        padding_needed_cols = col_divisor - (N % col_divisor)
    
    # Return original tensor if no padding is needed
    if padding_needed_rows == 0 and padding_needed_cols == 0:
        return tensor
    
    # Pad the tensor using jax.numpy.pad
    # PyTorch pad format: (left, right, top, bottom)
    # JAX pad format: ((before_axis0, after_axis0), (before_axis1, after_axis1))
    padded_tensor = jnp.pad(
        tensor,
        ((0, padding_needed_rows), (0, padding_needed_cols)),
        mode='constant',
        constant_values=0.0
    )
    
    return padded_tensor


def remove_pad(
    tensor: jnp.ndarray, 
    original_shape: tuple
) -> jnp.ndarray:
    """
    JAX equivalent of _rm_pad_tensor from PyTorch quantization code.
    Remove padding added by pad_tensor.
    
    Args:
        tensor: Padded tensor
        original_shape: Original shape before padding
        
    Returns:
        Tensor with padding removed
    """
    # Simply slice back to original dimensions
    return tensor[:original_shape[0], :original_shape[1]]


def validate_fp4_constraints(tensor: jnp.ndarray, kernel_type: str = "standard") -> bool:
    """
    Validate that tensor meets FP4 kernel constraints.
    
    Args:
        tensor: Input tensor to validate
        kernel_type: Type of kernel ("standard" or "transpose")
        
    Returns:
        True if tensor meets constraints, False otherwise
    """
    if kernel_type == "standard":
        return tensor.size % 512 == 0
    elif kernel_type == "transpose":
        if len(tensor.shape) != 2:
            return False
        num_rows, num_cols = tensor.shape
        return (num_cols % 64 == 0) and (num_rows % 128 == 0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

