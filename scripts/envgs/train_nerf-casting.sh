# Training script for nerf-casting dataset
# Including scenes: compact, grinder, hatchback, toaster

evc-train -c configs/exps/envgs/nerf-casting/envgs_compact.yaml exp_name=envgs/nerf-casting/envgs_compact
evc-train -c configs/exps/envgs/nerf-casting/envgs_grinder.yaml exp_name=envgs/nerf-casting/envgs_grinder
evc-train -c configs/exps/envgs/nerf-casting/envgs_hatchback.yaml exp_name=envgs/nerf-casting/envgs_hatchback
evc-train -c configs/exps/envgs/nerf-casting/envgs_toaster.yaml exp_name=envgs/nerf-casting/envgs_toaster
