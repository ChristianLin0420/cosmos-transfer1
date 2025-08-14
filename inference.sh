#!/bin/bash

# ==========================================
#   Cosmos-Transfer1 Inference Script
# ==========================================

# Set CUDA devices (default: all 8 GPUs)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Set checkpoint directory (default: ./checkpoints)
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# Number of GPUs to use (default: 8)
export NUM_GPU="${NUM_GPU:-8}"

# Output directory for generated videos
OUTPUT_DIR="outputs/example1_single_control_edge_upsampled_prompt"

# ControlNet configuration
CONTROLNET_SPECS="assets/inference_cosmos_transfer1_single_control_edge_short_prompt.json"

# Run inference
echo "=========================================="
echo "  Cosmos-Transfer1 Inference"
echo "  GPUs:           $CUDA_VISIBLE_DEVICES"
echo "  Checkpoints:    $CHECKPOINT_DIR"
echo "  Output Folder:  $OUTPUT_DIR"
echo "  ControlNet:     $CONTROLNET_SPECS"
echo "=========================================="

PYTHONPATH=$(pwd) torchrun \
    --nproc_per_node=$NUM_GPU \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    cosmos_transfer1/diffusion/inference/transfer.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --video_save_folder "$OUTPUT_DIR" \
        --controlnet_specs "$CONTROLNET_SPECS" \
        --offload_text_encoder_model \
        --upsample_prompt \
        --offload_prompt_upsampler \
        --offload_guardrail_models \
        --num_gpus $NUM_GPU

# End of script