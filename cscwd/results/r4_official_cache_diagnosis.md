# 官方 test 缓存贡献与历史控制诊断

- 版本：`r4_cache_diagnosis_v1`；仅使用已保存预测，不训练或重新前向。
- 四种交叉组合用于输出诊断；不作为重新训练的单输出消融。
- L2/AP由坐标和分数计分；本任务不重新计算热图AUC。

| 模型 | 坐标 | 状态分数 | L2 | AP | 准确稳定 | 状态错误 | gaze_shift响应 |
|---|---|---|---:|---:|---:|---:|---:|
| ordinary_gru | raw | raw | 0.098995 | 0.886801 | 0.527531 | 0.195165 | 0.582278 |
| ordinary_gru | refined | raw | 0.099482 | 0.886801 | 0.527480 | 0.195165 | 0.575949 |
| ordinary_gru | raw | refined | 0.098995 | 0.892941 | 0.509244 | 0.191038 | 0.563291 |
| ordinary_gru | refined | refined | 0.099482 | 0.892941 | 0.509093 | 0.191038 | 0.556962 |

## ordinary_gru 的窗口分解

```json
{
  "interpretation": "saved_output_numerical_decomposition; not_retrained_ablation_or_mechanism_causality",
  "eligible_windows": 19796,
  "accurate_stable_counts": {
    "M00": 10443,
    "M10": 10442,
    "M01": 10081,
    "M11": 10078
  },
  "accurate_stable_ratios": {
    "M00": 0.5275308143059204,
    "M10": 0.5274802990503132,
    "M01": 0.5092442917761164,
    "M11": 0.5090927460092948
  },
  "matrix_definition": {
    "M00": "raw_xy_raw_score",
    "M10": "refined_xy_raw_score",
    "M01": "raw_xy_refined_score",
    "M11": "refined_xy_refined_score"
  },
  "total_delta": {
    "window_count": -365,
    "ratio": -0.01843806829662558
  },
  "coordinate_effect_at_raw_score": {
    "window_count": -1,
    "ratio": -5.051525560719337e-05
  },
  "score_effect_at_raw_xy": {
    "window_count": -362,
    "ratio": -0.018286522529804002
  },
  "interaction": {
    "window_count": -2,
    "ratio": -0.00010103051121438674
  },
  "shapley": {
    "coordinate": {
      "window_count": -2.0,
      "ratio": -0.00010103051121438674
    },
    "score": {
      "window_count": -363.0,
      "ratio": -0.018337037785411195
    },
    "interpretation": "symmetric_average_of_the_two_output_replacement_orders; fractional_window_count_equivalents"
  },
  "window_transitions": {
    "lost": 632,
    "gained": 267,
    "both_accurate": 9811,
    "neither_accurate": 9086,
    "failure_mask_bits": {
      "1": "coordinate_unstable",
      "2": "localization_inaccurate",
      "4": "predicted_out_in_window"
    },
    "lost_refined_failure_masks": {
      "1": 208,
      "2": 4,
      "3": 28,
      "4": 364,
      "5": 20,
      "6": 2,
      "7": 6
    },
    "gained_raw_failure_masks": {
      "1": 209,
      "2": 1,
      "3": 27,
      "4": 30,
      "5": 0,
      "6": 0,
      "7": 0
    },
    "failure_mask_interpretation": "mutually_exclusive_bit_patterns; multiple_failed_conditions_can_overlap_in_one_window"
  }
}
```

| history_validity | raw | raw | 0.098995 | 0.886801 | 0.527531 | 0.195165 | 0.582278 |
| history_validity | refined | raw | 0.099244 | 0.886801 | 0.525308 | 0.195165 | 0.582278 |
| history_validity | raw | refined | 0.098995 | 0.892778 | 0.504748 | 0.193289 | 0.556962 |
| history_validity | refined | refined | 0.099244 | 0.892778 | 0.503334 | 0.193289 | 0.556962 |

## history_validity 的窗口分解

```json
{
  "interpretation": "saved_output_numerical_decomposition; not_retrained_ablation_or_mechanism_causality",
  "eligible_windows": 19796,
  "accurate_stable_counts": {
    "M00": 10443,
    "M10": 10399,
    "M01": 9992,
    "M11": 9964
  },
  "accurate_stable_ratios": {
    "M00": 0.5275308143059204,
    "M10": 0.5253081430592039,
    "M01": 0.5047484340270761,
    "M11": 0.5033340068700748
  },
  "matrix_definition": {
    "M00": "raw_xy_raw_score",
    "M10": "refined_xy_raw_score",
    "M01": "raw_xy_refined_score",
    "M11": "refined_xy_refined_score"
  },
  "total_delta": {
    "window_count": -479,
    "ratio": -0.024196807435845624
  },
  "coordinate_effect_at_raw_score": {
    "window_count": -44,
    "ratio": -0.002222671246716508
  },
  "score_effect_at_raw_xy": {
    "window_count": -451,
    "ratio": -0.02278238027884421
  },
  "interaction": {
    "window_count": 16,
    "ratio": 0.0008082440897150939
  },
  "shapley": {
    "coordinate": {
      "window_count": -36.0,
      "ratio": -0.0018185492018589614
    },
    "score": {
      "window_count": -443.0,
      "ratio": -0.022378258233986665
    },
    "interpretation": "symmetric_average_of_the_two_output_replacement_orders; fractional_window_count_equivalents"
  },
  "window_transitions": {
    "lost": 631,
    "gained": 152,
    "both_accurate": 9812,
    "neither_accurate": 9201,
    "failure_mask_bits": {
      "1": "coordinate_unstable",
      "2": "localization_inaccurate",
      "4": "predicted_out_in_window"
    },
    "lost_refined_failure_masks": {
      "1": 144,
      "2": 3,
      "3": 20,
      "4": 431,
      "5": 26,
      "6": 0,
      "7": 7
    },
    "gained_raw_failure_masks": {
      "1": 126,
      "2": 0,
      "3": 13,
      "4": 13,
      "5": 0,
      "6": 0,
      "7": 0
    },
    "failure_mask_interpretation": "mutually_exclusive_bit_patterns; multiple_failed_conditions_can_overlap_in_one_window"
  }
}
```

## 历史控制量

| 离线分组 | 帧数 | 历史可用帧 | 有历史时 validity均值 | 有历史时低/高validity占比 | refresh均值 |
|---|---:|---:|---:|---|---:|
| all | 31978 | 31680 | 0.671961 | 0.000000 / 0.000000 | 0.610501 |
| GT_inframe | 20655 | 20476 | 0.657332 | 0.000000 / 0.000000 | 0.605228 |
| GT_out_of_frame | 11323 | 11204 | 0.698697 | 0.000000 / 0.000000 | 0.620120 |
| GT_stable_inframe_pair_endpoints | 19603 | 19603 | 0.656590 | 0.000000 / 0.000000 | 0.604927 |
| near_enter_frame | 1733 | 1726 | 0.676558 | 0.000000 / 0.000000 | 0.610681 |
| near_leave_frame | 1654 | 1647 | 0.673051 | 0.000000 / 0.000000 | 0.609823 |
| near_gaze_shift | 1194 | 1188 | 0.667242 | 0.000000 / 0.000000 | 0.609447 |
| history_available | 31680 | 31680 | 0.671961 | 0.000000 / 0.000000 | 0.610451 |
| no_history | 298 | 0 | null | null / null | 0.615787 |

低/高validity阈值固定为0.1/0.9，分母只含历史可用帧；refresh均值按该组全部帧汇总。
完整分布、对应距离和修正幅度关联见JSON的 `models.history_validity.history_controls`。
事件邻域使用标签仅作离线分组，可重叠；控制量相关性不当作历史机制的因果证据。

下一步依据窗口分解与门控统计决定一次针对性调整；所有源缓存保持只读。
