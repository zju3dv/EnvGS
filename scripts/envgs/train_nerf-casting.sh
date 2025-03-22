# Training script for nerf-casting dataset
# Including scenes: compact, grinder, hatchback, toaster

ecb-train -c configs/exps/envgs/nerf-casting/envgs_compact.yaml exp_name=envgs/nerf-casting/envgs_compact
ecb-train -c configs/exps/envgs/nerf-casting/envgs_grinder.yaml exp_name=envgs/nerf-casting/envgs_grinder
ecb-train -c configs/exps/envgs/nerf-casting/envgs_hatchback.yaml exp_name=envgs/nerf-casting/envgs_hatchback
ecb-train -c configs/exps/envgs/nerf-casting/envgs_toaster.yaml exp_name=envgs/nerf-casting/envgs_toaster
