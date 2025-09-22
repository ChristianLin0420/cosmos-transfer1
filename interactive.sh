ACCOUNT="edgeai_tao-ptm_image-foundation-model-clip"
PARTITION="interactive_singlenode"
srun \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --job-name "gpu_interactive" \
    --gpus 8 \
    --ntasks-per-node 8 \
    --time 04:00:00 \
    --container-image="christianlin0420/cosmos-transfer1:latest" \
    --container-mounts=$HOME:/root,/lustre:/lustre,/lustre/fsw/portfolios/edgeai/users/chrislin/projects/cosmos-transfer1:/workspace/cosmos-transfer1,$HOME/.cache:/root/.cache \
    --pty /bin/bash

