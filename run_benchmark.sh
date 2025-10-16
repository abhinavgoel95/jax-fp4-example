#! /bin/bash
set -x

nvidia-smi

LOG_DIR="./"
NSYS_OUTPUT_FILE="${LOG_DIR}/output-nsys-profile"
XLA_DUMP_DIR="${LOG_DIR}/xla_dump"

export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_dump_to=${XLA_DUMP_DIR}
                --xla_gpu_enable_command_buffer=
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_exhaustive_tiling_search=true"

nsys version

nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --cuda-graph-trace=node python3 -u jax-example.py

set +x
