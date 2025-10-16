import jax
import jax.numpy as jnp
import numpy as np
import time
import os

from contextlib import contextmanager
from transformer_engine.jax.quantize import (
    QuantizerFactory,
)
from transformer_engine.common import recipe
from transformer_engine.jax.dense import dense

@contextmanager
def transformer_engine_context():
  """ If TransformerEngine is available, this context manager will provide the library with MaxText-specific details needed for correct operation. """
  try:
    from transformer_engine.jax.sharding import global_shard_guard, MeshResource
    # Inform TransformerEngine of MaxText's physical mesh resources.
    mesh_resource = MeshResource(
      dp_resource = None,
      tp_resource = None,
      fsdp_resource = "data",
      pp_resource = None,
      cp_resource = None,
    )
    with global_shard_guard(mesh_resource):
      yield
  except ImportError:
    yield

# Initialize JAX distributed training
def initialize_jax_distributed():
    """ Initialize JAX for multi-GPU distributed training"""
    # Set up JAX distributed
    if 'JAX_COORDINATOR_ADDRESS' in os.environ:
        jax.distributed.initialize()

    print(f"JAX process count: {jax.process_count()}")
    print(f"JAX process index: {jax.process_index()}")
    print(f"JAX local device count: {jax.local_device_count()}")
    print(f"JAX devices: {jax.devices()}")

# Initialize distributed training
initialize_jax_distributed()

key = jax.random.PRNGKey(1234)

# Define sharding for FSDP
def create_fsdp_sharding():
    """ Create sharding pattern for FSDP across 8 GPUs"""
    devices = jax.devices()
    if len(devices) >= 8:
        # Shard across 8 GPUs
        mesh = jax.sharding.Mesh(devices[:8], ('data',))
        # Shard batch dimension across data axis
        sharding = jax.sharding.PartitionSpec('data', None, None)
        return mesh, sharding
    else:
        # Fallback to single device
        mesh = jax.sharding.Mesh(devices, ('data',))
        sharding = jax.sharding.PartitionSpec(None, None, None)
        return mesh, sharding

# Create sharding
mesh, sharding = create_fsdp_sharding()

@contextmanager
def transformer_engine_context():
  """ If TransformerEngine is available, this context manager will provide the library with MaxText-specific details needed for correcct operation. """
  try:
    from transformer_engine.jax.sharding import global_shard_guard, MeshResource
      
    # Inform TransformerEngine of MaxText's physical mesh resources.
    mesh_resource = MeshResource(
      dp_resource = None,
      tp_resource = None,
      fsdp_resource = "data",
      pp_resource = None,
      cp_resource = None,
    )
    with global_shard_guard(mesh_resource):
      yield
  except ImportError:
    yield

@jax.jit
def fp4_dot(a, b):
    dim0, dim1, dim2 = a.shape
    reshaped_inputs = a.reshape(-1, dim2)
    reshaped_inputs = jax.lax.with_sharding_constraint(reshaped_inputs, jax.sharding.PartitionSpec("data", None))

    with transformer_engine_context():
        data_layout = "NN"
        contracting_dims = ((1,), (0,))
        quantizer_set = QuantizerFactory.create_set(fp8_recipe=recipe.NVFP4BlockScaling())
        
        dense_fn = lambda x, w: dense(x, w, bias=None, contracting_dims=contracting_dims, quantizer_set=quantizer_set)
        out = dense_fn(reshaped_inputs, b)

    return out

@jax.jit
def bf16_dot(a, b):
    dim0, dim1, dim2 = a.shape
    reshaped_inputs = a.reshape(-1, dim2)
    reshaped_inputs = jax.lax.with_sharding_constraint(reshaped_inputs, jax.sharding.PartitionSpec("data", None))
    out = reshaped_inputs @ b
    return out

B, M, N, K = 16, 8192, 14336, 4096
n_warmup = 2
n_benchmark = 5

print('jax.device_count(): ', jax.device_count())
print('jax.local_device_count(): ', jax.local_device_count())
print('jax.devices(): ', jax.devices())
print('jax.local_devices(): ', jax.local_devices())

# Create data with proper sharding for FSDP
def create_sharded_data():
    """Create data arrays with FSDP sharding"""
    
    # Create data on each device
    local_key = jax.random.fold_in(key, jax.process_index())

    A = jax.random.uniform(key=local_key, shape=(B, M, K), dtype=jnp.bfloat16)
    W = jax.random.uniform(key=local_key, shape=(K, N), dtype=jnp.bfloat16)
    grad = jax.random.normal(local_key, shape=(B*M, N)).astype(jnp.bfloat16)

    # Apply sharding constraints
    with mesh:
        A = jax.lax.with_sharding_constraint(A, sharding)
        W = jax.lax.with_sharding_constraint(W, jax.sharding.PartitionSpec('data', None))  
        grad = jax.lax.with_sharding_constraint(grad, jax.sharding.PartitionSpec('data', None))

    return A, W, grad

A, W, grad = create_sharded_data()

# Warm up 
with transformer_engine_context():
    with mesh:
        for _ in range(n_warmup):
            out, f_vjp = jax.vjp(fp4_dot, A, W)
            input_grad, params_grad = f_vjp(grad)

# Benchmark the function
start_time = time.time()
with transformer_engine_context():
    with mesh:
        for _ in range(n_benchmark):  # Run multiple times to get an average time
            out, f_vjp = jax.vjp(fp4_dot, A, W)
            input_grad, params_grad = f_vjp(grad)
jax.block_until_ready(input_grad)
end_time = time.time()

# Calculate average execution time
average_time = (end_time - start_time) / n_benchmark
print(f"Average execution time for FP4 GEMM: {average_time * 1e3:.3f} ms")

# BF16 reference
# Warm up 
with mesh:
    for _ in range(n_warmup):
        out, f_vjp = jax.vjp(bf16_dot, A, W)
        input_grad, params_grad = f_vjp(grad)
        
# Benchmark the function
start_time = time.time()
with mesh:
    for _ in range(n_benchmark):  # Run multiple times to get an average time
        out, f_vjp = jax.vjp(bf16_dot, A, W)
        input_grad, params_grad = f_vjp(grad)
jax.block_until_ready(out)
end_time = time.time()

# Calculate average execution time
average_time = (end_time - start_time) / n_benchmark
print(f"Average execution time for BF16 GEMM: {average_time * 1e3:.3f} ms")
