conda activate mappo-sc
find ./ -type d | grep -E "wandb/run|wandb/offline" | grep -v files | grep -v logs | xargs wandb sync --no-mark-synced