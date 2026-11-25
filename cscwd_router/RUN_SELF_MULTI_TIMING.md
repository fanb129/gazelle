# 当前剩余GPU实验：仅自身模型的多人效率

用户已授权由助手在GPU0启动。只比较同一VAT-B选中权重的共享K100、逐人独立K50、max-union共享K50；N=1/2/3嵌套真实GT头框。原9个精度单元和四组单人效率保留，不重跑精度、不训练、不再测官方GazeLLE。

每个N跑正反序，各为独立进程，共6个单元/18组测量。FP16、batch_images=1、seed0、50 warmup、200×5同步延迟、500×5吞吐。完整forward含图像复制（独立路径）、backbone、Router、gather/scatter、回填与decoder；读取、预处理、H2D、加载checkpoint在外。记录真实block batch/tokens、GPU UUID、ms/frame、images/s、queries/s、peak allocated MiB。每轮保留独占检查，发现其他compute PID即退出。

建议协调连续1小时独占3090；30–45分钟为基于既有单人耗时的估算，不是本次实测。开始前与每轮检查卡占用，禁止与训练并行。

固定测速worktree：`/home/fb/src/paper/gazelleV1_self_multi_timing`。由助手经Git部署并CPU预检，运行期间不更新源码。

完整启动命令（助手执行一次；不要重复启动）：

```bash
cd /home/fb/src/paper/gazelleV1_self_multi_timing
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_self_multi_timing/cscwd_router/code/scripts/run_self_multi_timing.py \
  --source-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/self_multi_timing_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --device cuda --precision fp16 --warmup 50 --latency-iters 200 \
  --throughput-iters 500 --repeats 5 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/self_multi_timing_gpu0.log 2>&1 < /dev/null &
echo $! > /home/fb/src/paper/gazelleV1/logs/cscwd_router/self_multi_timing_gpu0.pid
```

输出为 `results/self_multi_timing_seed0/n{1,2,3}_{forward,reverse}.json` 和 `summary.json/md`。没有大输出或新checkpoint。完成后由助手核对六单元、同GPU/同权重、正反序、嵌套查询、token trace和汇总，再经Git回传小结果。独立K50与union K50可对应已完成多人精度；本入口不生成K100多人精度。

CPU预检：在同一命令加 `--check-only`，去掉nohup、setsid、后台与重定向，并使用 `CUDA_VISIBLE_DEVICES=""`。预检只读权重/数据/图像/源码，不执行模型forward或计时。

VAT-L单人效率已完成：K100/K50的B1为39.4368/38.7860ms，B8为181.8345/138.5573ms，相对自身K100加速1.0168/1.3123倍；见 [审查报告](results/E3_VATL_EFFICIENCY_REVIEW.md)。按当前收口范围，剩余GPU任务就是本多人计时。
