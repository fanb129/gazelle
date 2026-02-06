## gazefollow b
CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb_coarse_to_fine" > train_gazefollow_vitb_coarse_to_fine.log 2>&1 &
## gazefollow l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitl16" --exp_name="train_gazefollow_vitl_coarse_to_fine" > train_gazefollow_vitl_coarse_to_fine.log 2>&1 &

## vat b
CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_vitb" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb/2026-01-29_15-16-01/epoch_14.pt" > train_vat_vitb.log 2>&1 &
## vat l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitl16_inout" --exp_name="train_vat_vitl" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt" > train_vat_vitl.log 2>&1 &