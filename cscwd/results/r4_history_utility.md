# R4-D5 训练集历史候选效用

- Version: `r4_history_utility_v1`; official train only; no training or test access.
- Original InOut is fixed. Candidate AUC is not computed; only coordinates are stored.
- Coverage: `{"tracks": 1033, "person_frames": 132563, "localization_frames": 88919, "history_available_frames": 131528, "utility_frames": 88270, "group_counts": {"start_or_gap": 1035, "gt_out": 43258, "enter_frame": 954, "stable_pair": 83901, "intermediate_motion": 2306, "gaze_shift": 1109}}`.
- Raw train InOut AP (unchanged): `0.9583460099473562`.
- Utility denominator: current GT in-frame and contiguous past available; correctness radius 0.1.

| Candidate | Train L2 | Both correct | Current only | Candidate only (rescue) | Both wrong | Rescue blocked by raw out | Accurate stable | Gaze responses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 0.09431258271942627 | 65439 | 0 | 0 | 22831 | 0 | 0.5560661550304415 | 499 |
| current_grid | 0.09401869566366605 | 64846 | 593 | 594 | 22237 | 37 | 0.5990809209452052 | 504 |
| previous_raw | 0.09580982525855716 | 63663 | 1776 | 1421 | 21410 | 81 | 0.5542699853040659 | 490 |
| ema_075 | 0.09433170985450452 | 64999 | 440 | 408 | 22423 | 24 | 0.5740978329328885 | 502 |
| transport_previous | 0.10168261175691731 | 62294 | 3145 | 2624 | 20207 | 124 | 0.5203643658587791 | 495 |

## Utility by offline GT group

| Candidate | Group | Frames | Candidate only (rescue) | Current only (harm) |
|---|---|---:|---:|---:|
| current | start_or_gap | 0 | 0 | 0 |
| current | gt_out | 0 | 0 | 0 |
| current | enter_frame | 954 | 0 | 0 |
| current | stable_pair | 83901 | 0 | 0 |
| current | intermediate_motion | 2306 | 0 | 0 |
| current | gaze_shift | 1109 | 0 | 0 |
| current_grid | start_or_gap | 0 | 0 | 0 |
| current_grid | gt_out | 0 | 0 | 0 |
| current_grid | enter_frame | 954 | 7 | 18 |
| current_grid | stable_pair | 83901 | 538 | 547 |
| current_grid | intermediate_motion | 2306 | 43 | 25 |
| current_grid | gaze_shift | 1109 | 6 | 3 |
| previous_raw | start_or_gap | 0 | 0 | 0 |
| previous_raw | gt_out | 0 | 0 | 0 |
| previous_raw | enter_frame | 954 | 18 | 81 |
| previous_raw | stable_pair | 83901 | 1292 | 1464 |
| previous_raw | intermediate_motion | 2306 | 101 | 146 |
| previous_raw | gaze_shift | 1109 | 10 | 85 |
| ema_075 | start_or_gap | 0 | 0 | 0 |
| ema_075 | gt_out | 0 | 0 | 0 |
| ema_075 | enter_frame | 954 | 1 | 17 |
| ema_075 | stable_pair | 83901 | 381 | 358 |
| ema_075 | intermediate_motion | 2306 | 25 | 47 |
| ema_075 | gaze_shift | 1109 | 1 | 18 |
| transport_previous | start_or_gap | 0 | 0 | 0 |
| transport_previous | gt_out | 0 | 0 | 0 |
| transport_previous | enter_frame | 954 | 22 | 93 |
| transport_previous | stable_pair | 83901 | 2456 | 2772 |
| transport_previous | intermediate_motion | 2306 | 127 | 186 |
| transport_previous | gaze_shift | 1109 | 19 | 94 |

## GT-informed fixed-set reference

```json
{
  "interpretation": "GT_informed_fixed_candidate_set_reference; not_deployable_or_all_video_information_upper_bound",
  "candidate_names": [
    "current",
    "previous_raw",
    "ema_075",
    "transport_previous"
  ],
  "tie_break": "current_then_fixed_ORACLE_NAMES_order",
  "localization_frames": 88919,
  "l2": 0.08165971810370352,
  "current_l2": 0.09431258271942627,
  "current_wrong_frames": 23024,
  "rescued_current_errors": 3126,
  "choice_counts": {
    "current": 55817,
    "previous_raw": 9193,
    "ema_075": 597,
    "transport_previous": 23312
  },
  "utility_history_available": {
    "frames": 88270,
    "current_wrong_frames": 22831,
    "rescued_current_errors": 3126,
    "fixed_inout_joint_rescues": 2972,
    "rescue_blocked_by_raw_out": 154
  }
}
```

Oracle choices are scoring only; no GT/oracle result enters candidate memory.
This reference is not a deployable model or a bound on all video information.
Detailed coordinates, groups, errors and event records remain on the server.
