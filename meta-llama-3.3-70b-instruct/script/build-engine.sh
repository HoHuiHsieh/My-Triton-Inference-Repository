# !/bin/bash
# This script is used to build the engine for the model using TensorRT.
# It converts the checkpoint to a format suitable for TensorRT and then builds the engine.
# Ensure you have the required environment variables set before running this script.
# Usage: ./build-engine.sh

# Quantize the model into FP8 precision with post-training method
python ./src/quantize.py \
    --model_dir ./model \
    --output_dir ./checkpoint/FP8 \
    --dtype bfloat16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --calib_size 512

# Convert the checkpoint to TensorRT format
trtllm-build --checkpoint_dir ./checkpoint/FP8 \
    --output_dir ./engine/FP8 \
    --gemm_plugin auto \
    --max_batch_size 16 \
    --max_input_len 4096 \
    --kv_cache_type paged \
    --gemm_swiglu_plugin fp8 \
    --fp8_rowwise_gemm_plugin auto \
    --low_latency_gemm_plugin fp8 \
    --low_latency_gemm_swiglu_plugin fp8 \
    --use_fp8_context_fmha enable
