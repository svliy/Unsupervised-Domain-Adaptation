CUDA_VISIBLE_DEVICES=3 python train.py --config_name bp4d2gft

CUDA_VISIBLE_DEVICES=2 python train_target.py --config_name bp4d2disfa
CUDA_VISIBLE_DEVICES=1 python train_target.py --config_name bp4d2gft

CUDA_VISIBLE_DEVICES=1 python train_target.py --config_name bp4d


CUDA_VISIBLE_DEVICES=1 nohup python -u train_target.py --config_name bp4d2gft > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_target.py --config_name bp4d2disfa > /dev/null 2>&1 &

python train_target.py --config_name bp4d2gft