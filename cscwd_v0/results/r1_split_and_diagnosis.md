# R1 开发集分组与单帧诊断

机器准备完成；人物/目标诊断尚未人工填写，R1 阶段尚未验收。

- 分组：`conservative_annotation_video_directory_group; not_verified_independent_recording_id`。
- 配置：seed `20261117`，validation 组比例 `0.2`（请求值）。

| 集合 | groups | tracks | 人物帧 | 图像帧 | AUC | L2 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 32 | 827 | 98362 | 44839 | 0.9435150747270145 | 0.09907750524229125 | 0.9579390627184529 |
| validation | 8 | 206 | 34201 | 13668 | 0.9542055630729601 | 0.07960456904362263 | 0.9609226821493625 |

- train groups：`["A Play With Words", "All in the Family", "Arrested Development", "Band of Brothers", "Before Sunrise", "Big Bang Theory", "Breaking Bad", "Cheers", "Conan", "Coveted", "Crazy Rich Asian", "Driving Miss Daisy", "Friends", "Give Me One Reason", "Gone with the Wind", "Hearing", "Interview with Bill Gates", "Jersey Shore", "Keeping Up With the Kardashians", "Modern Family", "My Dinner with Andre", "Orange is the New Black", "Project Runway", "Secret", "Seinfeld", "Silicon Valley", "Sound of Music", "Star Wars", "Suits", "Tartuffe", "The View", "Three Idiots"]`。
- validation groups：`["BTS at Jimmy Fallon", "Grey's Anatomy", "How I Met Your Mother", "Interview at the Oscars", "Sherlock", "The Ellen Show", "UFC Octagon Interview", "Veep"]`。

| 集合 | 准确稳定占比 | 状态错误率 | 稳定 GT 状态翻转率 | 状态响应率 | joint gaze_shift 响应率 |
|---|---:|---:|---:|---:|---:|
| train | 0.5410856072683372 | 0.13012138834102602 | 0.019573546560053307 | 0.7234513274336283 | 0.611764705882353 |
| validation | 0.6018742013346585 | 0.11125405689892108 | 0.017669105836159202 | 0.7597765363128491 | 0.6108597285067874 |

## 训练组诊断抽样

代理事件不自动标注为真实换目标。每例 review_status 保持 unreviewed，回传后查看 RGB 与预测再填写。

| 类别 | 候选 | 独立 clip | 选中 |
|---|---:|---:|---:|
| gaze_shift_proxy | 817 | 169 | 3 |
| state_transition | 1485 | 314 | 3 |
| prediction_jump | 2002 | 206 | 3 |
| stable_reference | 2415 | 347 | 3 |

## 小预览

### gaze_shift_proxy_86e2404b163a12da

![frames](<r1_previews/gaze_shift_proxy_86e2404b163a12da.png>)

### gaze_shift_proxy_ad7d0f44fb05f33e

![frames](<r1_previews/gaze_shift_proxy_ad7d0f44fb05f33e.png>)

### gaze_shift_proxy_cc69d20ac0a17390

![frames](<r1_previews/gaze_shift_proxy_cc69d20ac0a17390.png>)

### state_transition_1a55a51014a2d4c5

![frames](<r1_previews/state_transition_1a55a51014a2d4c5.png>)

### state_transition_4a3687abe413658e

![frames](<r1_previews/state_transition_4a3687abe413658e.png>)

### state_transition_9c4a0e403ec68088

![frames](<r1_previews/state_transition_9c4a0e403ec68088.png>)

### prediction_jump_172a63ed65192eb1

![frames](<r1_previews/prediction_jump_172a63ed65192eb1.png>)

### prediction_jump_d6a0329da386a97b

![frames](<r1_previews/prediction_jump_d6a0329da386a97b.png>)

### prediction_jump_e09f32f09431c9a4

![frames](<r1_previews/prediction_jump_e09f32f09431c9a4.png>)

### stable_reference_f7dc273e7cb77301

![frames](<r1_previews/stable_reference_f7dc273e7cb77301.png>)

### stable_reference_002728e621e28b64

![frames](<r1_previews/stable_reference_002728e621e28b64.png>)

### stable_reference_2beabfdb309a8d1f

![frames](<r1_previews/stable_reference_2beabfdb309a8d1f.png>)

## 协议与后续

旧 joint 时序协议与独立 state_temporal_v1 分列。AP 使用连续分数；状态窗口不被画内坐标跳变截断。
完整分组/选择配置/候选诊断/逐事件数据留服务器，路径及 SHA256 见 JSON。
下一步：本地核对分组和来源，查看所选训练片段并记录诊断，再验收 R1；尚未开始 R2 或模型训练。
