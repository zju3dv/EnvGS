# Training script for ref_real dataset
# Including scenes: sedan, spheres, toycar

ecb-train -c configs/exps/envgs/ref_real/envgs_sedan.yaml exp_name=envgs/ref_real/envgs_sedan
ecb-train -c configs/exps/envgs/ref_real/envgs_spheres.yaml exp_name=envgs/ref_real/envgs_spheres
ecb-train -c configs/exps/envgs/ref_real/envgs_toycar.yaml exp_name=envgs/ref_real/envgs_toycar
