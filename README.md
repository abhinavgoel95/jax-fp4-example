## JAX FP4 GeMM Example


### Set up:

Container: `ghcr.io/nvidia/jax:maxtext-2025-08-20`

TE branch: `https://github.com/abhinavgoel95/TransformerEngine/tree/abgoel/nvfp4-kernel-wip` (abgoel/nvfp4-kernel-wip branch; remember to clone all submodules as well)

```
pip install pybind11 ninja

NVTE_CUDA_ARCHS=100a pip install --no-build-isolation -e </path/to/TE> -v

python jax-example.py

```
