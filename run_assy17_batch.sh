#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CHECKPOINT_DIR=./checkpoints
export NUM_GPU=8

# Create output directory
mkdir -p outputs/assy17_batch_experiments/

echo "Starting COMPREHENSIVE batch processing of assy17 configurations..."
echo "Using JSONL batch input for efficient processing"
echo "Base config: configs/assy17_batch_base.json"
echo "Batch input: batch_inputs/assy17_experiments.jsonl"
echo "Including blur strength experiments"

# Run the transfer with batch processing
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/assy17_batch_experiments/ \
    --controlnet_specs configs/assy17_batch_base.json \
    --batch_input_path batch_inputs/assy17_experiments.jsonl \
    --batch_size 1 \
    --blur_strength "medium" \
    --canny_threshold "medium" \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --fps 24 \
    --num_gpus $NUM_GPU

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Batch processing completed successfully!"
    echo "Results saved in outputs/assy17_batch_experiments/"
    echo "Each experiment will be in its own numbered folder:"
    echo "  - video_0: guidance_low (guidance=3.0)"
    echo "  - video_1: guidance_medium (guidance=5.0)" 
    echo "  - video_2: original_baseline (guidance=8.0)"
    echo "  - video_3: guidance_high (guidance=12.0)"
    echo "  - video_4: sigma_low (sigma_max=50.0)"
    echo "  - video_5: sigma_high (sigma_max=120.0)"
    echo "  - video_6: control_balanced (weights=0.5)"
    echo "  - video_7: control_depth_heavy (depth=1.0, seg=0.3)"
    echo "  - video_8: control_seg_heavy (depth=0.3, seg=1.0)"
    echo "  - video_9: blur_very_low (blur_strength=very_low)"
    echo "  - video_10: blur_low (blur_strength=low)"
    echo "  - video_11: blur_high (blur_strength=high)"
    echo "  - video_12: blur_very_high (blur_strength=very_high)"
    echo "=========================================="
else
    echo "❌ Batch processing failed!"
    exit 1
fi
