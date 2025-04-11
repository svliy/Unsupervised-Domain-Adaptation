# gft ---> bp4d

# cross: stage 1
CUDA_VISIBLE_DEVICES=1 python train_stage1.py --config_name gft2bp4d_stage1

# bp4d ---> gft
# source

# cross: stage 1
CUDA_VISIBLE_DEVICES=1 python train_stage1.py --config_name bp4d2gft
# cross: stage 2
CUDA_VISIBLE_DEVICES=1 python train_stage2.py --config_name bp4d2gft
# target
CUDA_VISIBLE_DEVICES=1 python train_target.py --config_name bp4d2gft

# run back
CUDA_VISIBLE_DEVICES=1 nohup python -u train_target.py --config_name bp4d2gft > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_target.py --config_name bp4d2disfa > /dev/null 2>&1 &