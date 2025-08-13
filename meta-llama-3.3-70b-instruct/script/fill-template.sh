# !/bin/bash
# This script is used to start the Triton Inference Server with the Llama 3.1 8B Instruct model.
# It sets up the environment, loads the model and engine directories, and runs the Triton server.
# Ensure you have the required environment variables set before running this script.
# Usage: ./run-tritonserver.sh

# Set the environment variables
export ROOT_DIR=/root/triton
export TOKENIZER_DIR=${ROOT_DIR}/tokenizer
export ENGINE_DIR=${ROOT_DIR}/engine
export REPO_DIR=./repo
export MAX_BATCH_SIZE=16
export FREE_GPU_MEM_FRACTION=0.7
export MAX_QUEUE_DELAY=100

# Update the repo directory to the latest version
rm -rf ${REPO_DIR}
cp -a ./raw-repository ${REPO_DIR}

# Fill in the template files with the appropriate values
echo "Filling in 'counter/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/counter/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${TOKENIZER_DIR},usageprocessing_instance_count:1

echo "Filling in 'llama-3.3-70b-instruct/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/llama-3.3-70b-instruct/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:TYPE_FP32

echo "Filling in 'postprocessing/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1

echo "Filling in 'preprocessing/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/preprocessing/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY},max_queue_size:${MAX_BATCH_SIZE},tokenizer_dir:${TOKENIZER_DIR},preprocessing_instance_count:1

echo "Filling in 'tensorrt_llm/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,max_queue_delay_microseconds:${MAX_QUEUE_DELAY},max_queue_size:${MAX_BATCH_SIZE},max_beam_width:1,engine_dir:${ENGINE_DIR},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:${FREE_GPU_MEM_FRACTION},exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32

echo "Filling in 'tokenizer/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/tokenizer/config.pbtxt tokenizer_dir:${TOKENIZER_DIR}