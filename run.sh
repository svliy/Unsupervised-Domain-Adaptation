# run back
CUDA_VISIBLE_DEVICES=1 nohup python -u train_target.py --config_file bp4d2gft > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_target.py --config_file bp4d2disfa > /dev/null 2>&1 &

# bp4d ---> gft
CUDA_VISIBLE_DEVICES=5 python train_stage1.py --config_file bp4d2gft_stage1
CUDA_VISIBLE_DEVICES=5 python train_stage2.py --config_file bp4d2gft_stage2
CUDA_VISIBLE_DEVICES=7 python train_target.py --config_file bp4d2gft_target # target

# gft ---> bp4d
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --config_file gft2bp4d_stage1
CUDA_VISIBLE_DEVICES=0 python train_stage2.py --config_file gft2bp4d_stage2
CUDA_VISIBLE_DEVICES=7 python train_target.py --config_file gft2bp4d_target

# bp4d+ ---> gft
CUDA_VISIBLE_DEVICES=2 python train_stage1.py --config_file bp4d+2gft_stage1
CUDA_VISIBLE_DEVICES=3 python train_stage2.py --config_file bp4d+2gft_stage2

# gft ---> bp4d+
CUDA_VISIBLE_DEVICES=3 python train_stage1.py --config_file gft2bp4d+_stage1
CUDA_VISIBLE_DEVICES=2 python train_stage2.py --config_file gft2bp4d+_stage2
CUDA_VISIBLE_DEVICES=3 python train_target.py --config_file gft2bp4d+_target

# bp4d ---> disfa
CUDA_VISIBLE_DEVICES=1 python train_stage1.py --config_file bp4d2disfa_stage1
CUDA_VISIBLE_DEVICES=0 python train_stage2.py --config_file bp4d2disfa_stage2
CUDA_VISIBLE_DEVICES=5 python train_target.py --config_file bp4d2disfa_target

# disfa ---> bp4d
CUDA_VISIBLE_DEVICES=5 python train_stage1.py --config_file disfa2bp4d_stage1
CUDA_VISIBLE_DEVICES=4 python train_stage2_one.py --config_file disfa2bp4d_stage2
CUDA_VISIBLE_DEVICES=5 python train_target.py --config_file disfa2bp4d_target

# disfa ---> bp4d+
CUDA_VISIBLE_DEVICES=0 python train_target.py --config_file disfa2bp4d+_target


# target
## disfa
CUDA_VISIBLE_DEVICES=4 python train_target.py --config_file disfa_target
CUDA_VISIBLE_DEVICES=4 python train_target.py --config_file bp4d_target

