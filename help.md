## gazefollow b
CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_baseline" > train_gazefollow_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_sasa" --use_sasa > train_gazefollow_sasa.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_sasa_ggsf" --use_sasa --use_ggsf > train_gazefollow_sasa_ggsf.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_ggsf" --use_ggsf > train_gazefollow_ggsf.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_sasa_ggsf_aux" --use_sasa --use_ggsf --use_aux > train_gazefollow_sasa_ggsf_aux.log 2>&1 &

## gazefollow l
CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_gazefollow.py --model="gazelle_dinov3_vitl16" --exp_name="train_gazefollow_vitl_sasa_ggsf" --use_sasa --use_ggsf > train_gazefollow_vitl_sasa_ggsf.log 2>&1 &

## vat b
CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_baseline" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt" > train_vat_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa/2026-02-12_02-45-09/epoch_14.pt" --use_sasa > train_vat_sasa.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt" --use_sasa --use_ggsf > train_vat_sasa_ggsf.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf_aux/2026-03-11_10-21-57/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux.log 2>&1 &


## vat l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitl16_inout" --exp_name="train_vat_vitl_sasa_ggsf" --init_ckpt="" --use_sasa --use_ggsf > train_vat_vitl_sasa_ggsf.log 2>&1 &



v0

CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_gazefollow_v0.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb_v0" > train_gazefollow_vitb_v0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat_v0.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_vitb_v0" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt" > train_vat_vitb_v0.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_0" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_1" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa/2026-02-12_02-45-09/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_2" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_2.log 2>&1 &

