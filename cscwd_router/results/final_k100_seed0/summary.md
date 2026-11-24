# Selected-weight K50 versus K100 accuracy

FP32; identical selected weights and original test annotations. GF uses Avg L2 below. No training or checkpoint reselection.

| Dataset | Size | K50 AUC | K100 AUC | K50 L2 | K100 L2 | K50 AP | K100 AP |
|---|---|---|---|---|---|---|---|
| gazefollow | ViT-B | 0.956236 | 0.957218 | 0.101550 | 0.100484 | — | — |
| gazefollow | ViT-L | 0.960007 | 0.961650 | 0.094061 | 0.093095 | — | — |
| vat | ViT-B | 0.940033 | 0.940170 | 0.100131 | 0.100636 | 0.883183 | 0.878522 |
| vat | ViT-L | 0.948264 | 0.947314 | 0.089676 | 0.090269 | 0.891592 | 0.898700 |

K100 traces: 1024 retained patches, 1029 tokens at every block; 12 blocks for B and 24 for L. GF: 4,782 queries; VAT: 31,978 queries (20,655 in-frame).

Artifacts remain on the server; paths and SHA256 are registered in RESULT_PROVENANCE.md.
