## JAX FP4 GeMM Example


### Set up:

TE branch: `https://github.com/abhinavgoel95/TransformerEngine/tree/abgoel/nvfp4-kernel-wip` (remember to clone all submodules as well)

```
pip install pybind11 ninja

NVTE_CUDA_ARCHS=100a pip install --no-build-isolation -e </path/to/TE> -v

python jax-example.py

```
