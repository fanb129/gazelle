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

CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_ggsf" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_ggsf/2026-02-28_15-17-39/epoch_14.pt" --use_ggsf > train_vat_ggsf.log 2>&1 &


## vat l
CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitl16_inout" --exp_name="train_vat_vitl_sasa_ggsf" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitl_sasa_ggsf/2026-03-22_13-49-56/epoch_14.pt" --use_sasa --use_ggsf > train_vat_vitl_sasa_ggsf.log 2>&1 &



v0

CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_gazefollow_v0.py --model="gazelle_dinov3_vitb16" --exp_name="train_gazefollow_vitb_v0" > train_gazefollow_vitb_v0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat_v0.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_vitb_v0" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt" > train_vat_vitb_v0.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_0" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_1" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa/2026-02-12_02-45-09/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 setsid nohup python -u scripts/train_vat.py --model="gazelle_dinov3_vitb16_inout" --exp_name="train_vat_sasa_ggsf_aux_2" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt" --use_sasa --use_ggsf --use_aux > train_vat_sasa_ggsf_aux_2.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/generate_comparisons.py \
    --input_dir "/newhome/fb/dataset/gazefollow_extended/test2/00000000" \
    --output_dir "/home/fb/src/paper/gazelleV1/visualizations_paper" \
    --json_path "/newhome/fb/dataset/gazefollow_extended/test_preprocessed.json" \
    --base_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt" \
    --spot_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt" > generate_comparisons.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python scripts/generate_comparisons.py \
    --input_dir "/newhome/fb/dataset/gazefollow_extended/test2/00000000" \
    --output_dir "/home/fb/src/paper/gazelleV1/visualizations_paper" \
    --json_path "/newhome/fb/dataset/gazefollow_extended/test_preprocessed.json" \
    --base_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt" \
    --spot_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt"

CUDA_VISIBLE_DEVICES=0 python scripts/generate_comparisons.py \
    --input_dir "/newhome/fb/dataset/gazefollow_extended/test2" \
    --output_dir "/home/fb/src/paper/gazelleV1/visualizations_paper" \
    --json_path "/newhome/fb/dataset/gazefollow_extended/test_preprocessed/test_near.json" \
    --base_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt" \
    --spot_ckpt "/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt"


CUDA_VISIBLE_DEVICES=0 python scripts/eval_gazefollow.py

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python -u scripts/eval_vat.py \
--json_path "/newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_lt3.json" \
--vis_dir "/newhome/fb/dataset/videoattentiontarget/exp_vis/test_crowd" > eval_vat_test_crowd_lt3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python -u scripts/eval_vat_dinov2.py \
--json_path "/newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_4.json" \
--vis_dir "/newhome/fb/dataset/videoattentiontarget/exp_vis/test_crowd" > eval_vat_dinov2_test_crowd_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python -u scripts/eval_vat.py \
--json_path "/newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json" \
--vis_dir "/newhome/fb/dataset/videoattentiontarget/exp_vis/test_far" > eval_vat_test_far.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python -u scripts/eval_vat.py \
--json_path "/newhome/fb/dataset/videoattentiontarget/test_preprocessed.json" \
--vis_dir "/newhome/fb/dataset/videoattentiontarget/exp_vis/test_near" > eval_vat.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python -u scripts/eval_vat.py \
--json_path "/newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json" \
--vis_dir "/newhome/fb/dataset/videoattentiontarget/exp_vis/test_mixed_hard" > eval_vat_test_mixed_hard.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_gazefollow_dinov2.py --model="gazelle_dinov2_vitb14" --exp_name="train_gazefollow_dinov2_sasa_ggsf" --use_sasa --use_ggsf > train_gazefollow_dinov2_sasa_ggsf.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup python -u scripts/train_vat_dinov2.py --model="gazelle_dinov2_vitb14_inout" --exp_name="train_vat_dinov2_sasa_ggsf" --init_ckpt="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_dinov2_sasa_ggsf/2026-03-26_00-58-25/epoch_14.pt" --use_sasa --use_ggsf > train_vat_dinov2_sasa_ggsf.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup /home/fb/anaconda3/envs/py310/bin/python scripts/eval_gooreal.py \
  --data_path /newhome/fb/dataset/gooreal_data \
  --json_path /newhome/fb/dataset/gooreal_data/gooreal_test_preprocessed.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --batch_size 64 \
  --output rebuttal/results/p0/gooreal_eval1.json \
  --csv_output rebuttal/results/p0/gooreal_eval1.csv > gooreal_eval1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup /home/fb/anaconda3/envs/py310/bin/python scripts/eval_gooreal.py \
  --data_path /newhome/fb/dataset/gooreal_data \
  --json_path /newhome/fb/dataset/gooreal_data/gooreal_test_sparse_preprocessed.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --batch_size 64 \
  --output rebuttal/results/p0/gooreal_eval_test_sparse.json \
  --csv_output rebuttal/results/p0/gooreal_eval_test_sparse.csv > gooreal_eval_test_sparse.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 setsid nohup /home/fb/anaconda3/envs/py310/bin/python scripts/eval_bbox_noise.py \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --json_path /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_eq4.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --jitter_levels 0 5 10 20 \
  --seed 3106 \
  --batch_size 16 \
  --output rebuttal/results/p0/vat_crowd_bbox_noise_test_crowd_eq4.json \
  --csv_output rebuttal/results/p0/vat_crowd_bbox_noise_test_crowd_eq4.csv > vat_crowd_bbox_noise_test_crowd_eq4.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 setsid nohup /home/fb/anaconda3/envs/py310/bin/python scripts/compute_flops.py \
  --device cuda \
  --warmup_iters 50 \
  --measure_iters 200 \
  --batch_size 1 \
  --output rebuttal/results/p0/complexity_latency.json > complexity_latency.log 2>&1 &
