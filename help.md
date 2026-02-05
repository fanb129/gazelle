## gazefollow b
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb" > train_gazefollow_vitb.log 2>&1 &
## gazefollow l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitl16" --exp_name="train_gazefollow_vitl" > train_gazefollow_vitl.log 2>&1 &
## gazefollow b text
CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb_text" > train_gazefollow_vitb_text.log 2>&1 &
## gazefollow l text
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitl16" --exp_name="train_gazefollow_vitl_text" > train_gazefollow_vitl_text.log 2>&1 &

## gazefollow generate text
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com setsid nohup python -u data_prep/preprocess_gazefollow_text.py > data_prep/preprocess_gazefollow_text.log 2>&1 &
## vat generate text
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com setsid nohup python -u data_prep/preprocess_vat_text.py > data_prep/preprocess_vat_text.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb_text" --wandb="0" > train_gazefollow_vitb_text.log 2>&1 &