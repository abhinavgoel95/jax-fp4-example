## JAX FP4 GeMM Example


### Set up:

Container: `ghcr.io/nvidia/jax:maxtext-2025-10-15`

TE:  Use the latest TE from upstream `https://github.com/NVIDIA/TransformerEngine` (remember to clone all submodules as well)

```
pip install pybind11 ninja nvidia-mathdx

NVTE_CUDA_ARCHS=100a pip install --no-build-isolation -e </path/to/TE> -v

bash run_benchmark.sh
```
