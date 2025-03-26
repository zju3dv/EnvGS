# Training script for ref_real dataset
# Including scenes: sedan, spheres, toycar

evc-train -c configs/exps/envgs/ref_real/envgs_sedan.yaml exp_name=envgs/ref_real/envgs_sedan
evc-train -c configs/exps/envgs/ref_real/envgs_spheres.yaml exp_name=envgs/ref_real/envgs_spheres
evc-train -c configs/exps/envgs/ref_real/envgs_toycar.yaml exp_name=envgs/ref_real/envgs_toycar
