## gazefollow b
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov2_vitb14" --exp_name="train_gazefollow_vitb" > train_gazefollow_vitb.log 2>&1 &
## gazefollow l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov2_vitl14" --exp_name="train_gazefollow_vitl" > train_gazefollow_vitl.log 2>&1 &