## JAX FP4 GeMM Example


### Set up:

Container: `ghcr.io/nvidia/jax:maxtext-2025-08-20`

TE:  Use latest TE from upstream `https://github.com/NVIDIA/TransformerEngine` (remember to clone all submodules as well)

```
pip install pybind11 ninja

NVTE_CUDA_ARCHS=100a pip install --no-build-isolation -e </path/to/TE> -v

bash run_benchmark.sh
```
