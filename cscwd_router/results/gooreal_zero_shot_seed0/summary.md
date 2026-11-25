# GOO-Real fixed-checkpoint zero-shot evaluation

Full 2,146-image official test split; GF-trained weights; real GT head boxes; FP32; seed 0.

| Model | AUC ↑ | L2 ↓ | GT point coverage ↑ |
|---|---|---|---|
| Official GazeLLE-B (re-evaluated) | 0.901678 | 0.172190 | — |
| Router-B K50 | 0.896435 | 0.178415 | 0.928705 |
| Official GazeLLE-L (re-evaluated) | 0.900474 | 0.173867 | — |
| Router-L K50 | 0.895270 | 0.173924 | 0.926375 |

4 artifact pairs hashed; identical 2146 query/GT records; raw means matched within 1e-12

Published GazeLLE numbers use its original evaluation preparation; distinguish them from these matched-box re-evaluations.
