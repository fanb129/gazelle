# R4-D6 cost-supervised history selector

- Hard choice: current_grid or one-hop visual history; original InOut unchanged.
- Complete official train, five fresh epochs; complete test every epoch.
- Test used for epoch selection: AUC descending, L2 ascending, AP descending, earlier epoch.
- Selected epoch: 1; parameters: 296129.
- This is not an independent held-out estimate. No GT/oracle result feeds candidate memory.

| Model | Epoch | AUC | L2 | AP | Accurate stable | Gaze responses | History choices |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 0 | 0.940181658 | 0.098994832 | 0.886801040 | 0.5275308143059204 | 92 | 0 |
| current_grid | 0 | 0.942892323 | 0.097854149 | 0.886801040 | 0.5670842594463528 | 91 | 0 |
| ema_075 | 0 | 0.940119806 | 0.098941344 | 0.886801040 | 0.5425338452212568 | 90 | 0 |
| grid_ema_075 | 0 | 0.942837492 | 0.097946869 | 0.886801040 | 0.5798646191149727 | 90 | 0 |
| transport_previous | 0 | 0.937514158 | 0.105366569 | 0.886801040 | 0.5042432814710043 | 86 | 31978 |
| utility_selector | 1 | 0.942892323 | 0.097854149 | 0.886801040 | 0.5670842594463528 | 91 | 0 |
| utility_selector | 2 | 0.942857680 | 0.097945285 | 0.886801040 | 0.5665791068902809 | 91 | 188 |
| utility_selector | 3 | 0.942643594 | 0.098057759 | 0.886801040 | 0.5625884016973126 | 91 | 1156 |
| utility_selector | 4 | 0.942440353 | 0.098475784 | 0.886801040 | 0.5562739947464134 | 91 | 2908 |
| utility_selector | 5 | 0.942621986 | 0.098268459 | 0.886801040 | 0.5592543948272378 | 91 | 2087 |

All selected metrics come from one epoch. All five checkpoints and full outputs remain on server.
JSON preserves cost-loss totals, complete fixed controls, all-epoch metrics, deltas and paired events.
Compare stability and response jointly against current_grid and grid_ema_075 before attributing a history benefit.
