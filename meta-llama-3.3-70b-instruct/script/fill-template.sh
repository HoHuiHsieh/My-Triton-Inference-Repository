# !/bin/bash
# This script is used to start the Triton Inference Server with the Llama 3.1 8B Instruct model.
# It sets up the environment, loads the model and engine directories, and runs the Triton server.
# Ensure you have the required environment variables set before running this script.
# Usage: ./run-tritonserver.sh

export MODEL_DIR=${PWD}/model
export ENGINE_DIR=${PWD}/engine/FP8
export MAX_BATCH_SIZE=16
export GPU_MEM_FRACTION=0.9
export MAX_QUEUE_DELAY=100

# max_queue_delay_microseconds

rm -rf ./repository
cp -a ./raw-repository ./repository
python3 ./src/fill_template.py -i ./repository/preprocessing/config.pbtxt tokenizer_dir:${MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1 
python3 ./src/fill_template.py -i ./repository/postprocessing/config.pbtxt tokenizer_dir:${MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1 
python3 ./src/fill_template.py -i ./repository/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_DIR},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:${GPU_MEM_FRACTION},exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32 
python3 ./src/fill_template.py -i ./repository/llama-3.3-70b-instruct/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:TYPE_FP32 
python3 ./src/fill_template.py -i ./repository/usage_counter/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${MODEL_DIR},usageprocessing_instance_count:1 