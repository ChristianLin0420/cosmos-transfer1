# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=./checkpoints
export NUM_GPU=1

# Run the transfer
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/assy17_distilled/ \
    --controlnet_specs configs/assy17.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU