# Native Router CPU MACs

One image, one head query, 512x512. Conventional Conv/Linear/attention MACs; FLOPs = 2 MACs.
Untrained native weights; no quality or runtime result. Both paths retain the Router and the same multilayer decoder.

| Dataset | Size | K100 GMACs | K50 GMACs | MAC reduction |
|---|---|---|---|---|
| gazefollow | ViT-B | 112.890088 | 83.851873 | 25.72% |
| gazefollow | ViT-L | 369.304454 | 272.541861 | 26.20% |
| vat | ViT-B | 112.895628 | 83.857413 | 25.72% |
| vat | ViT-L | 369.309994 | 272.547400 | 26.20% |

Elementwise arithmetic, normalization, RoPE, softmax, sorting and memory movement are excluded from this MAC convention; their runtime remains included in the GPU forward benchmarks.
