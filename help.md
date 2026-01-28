## gazefollow b
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb" > train_gazefollow_vitb.log 2>&1 &
## gazefollow l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitl16" --exp_name="train_gazefollow_vitl" > train_gazefollow_vitl.log 2>&1 &