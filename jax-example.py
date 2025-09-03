import jax
import jax.numpy as jnp
import numpy as np
import jax.random

# Import Hadamard utilities
from hadamard_utils import HadamardTransform

# Import FP4 quantization utilities
from fp4_kernels import calculate_amax, pad_tensor, remove_pad, quantize_e2m1_scaled


key = jax.random.PRNGKey(3)
def rng_key():
    global key
    key, subkey = jax.random.split(key)
    return subkey

def test_e2m1_gemm(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    Alternative implementation using custom_vjp for more control over backward pass.
    custom_vjp allows you to define both forward and backward passes separately,
    which can be more intuitive for some operations.
    """
    
    qx, s_x, a_x = quantize_e2m1_scaled(x)
    qw, s_w, a_w = quantize_e2m1_scaled(jnp.expand_dims(jnp.transpose(w, axes=(1, 0)), axis=(0,)))

    partial_output = jax.nn.scaled_matmul(qx, qw, s_x, s_w)
    lhs_tensor_scale_inv = a_x / DTYPE_MAX
    rhs_tensor_scale_inv = a_w/ DTYPE_MAX
    output = partial_output * lhs_tensor_scale_inv * rhs_tensor_scale_inv
    return output.astype(jnp.bfloat16), (qx, qw)


# Initialize Hadamard transform
NH = 16
DATA_DTYPE_MAX = jnp.finfo(jnp.float4_e2m1fn).max.astype(jnp.float32)
SCALE_DTYPE_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
DTYPE_MAX = DATA_DTYPE_MAX * SCALE_DTYPE_MAX

if __name__ == "__main__":
    lhs = jax.random.normal(rng_key(), shape=(1, 2048, 2048)).astype(jnp.bfloat16)
    rhs = jax.random.normal(rng_key(), shape=(2048, 2048)).astype(jnp.bfloat16)
    gradient = jax.random.normal(rng_key(), shape=(lhs.shape[0], rhs.shape[0])).astype(jnp.bfloat16)
    out, _ = test_e2m1_gemm(lhs, rhs)
    print(f"x: {lhs.shape} {lhs.dtype} \n {lhs}")
    print(f"w: {rhs.shape} {rhs.dtype} \n {rhs}")
    print(f"out: {out.shape} {out.dtype} \n {out}")
    print(f"ref out: \n {jax.numpy.dot(lhs, rhs)}")


    
