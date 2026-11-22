"""Existing server artifacts inspected read-only on 2026-11-22."""

BACKBONE_SHA = {
    "b": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
    "l": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
}

INITIALIZERS = {
    ("gazefollow", "b"): {
        "path": "experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt",
        "sha256": "26010ecc293cd51d2ce69f95e8111ca46632e425e1463b22fb0980aacf67fbef",
        "kind": "decoder",
    },
    ("gazefollow", "l"): {
        "path": "experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt",
        "sha256": "7dbe0f2e07d003f3d8c479751dd08c3bbc642d8ab6558c11847a6e5d38fe58af",
        "kind": "decoder",
    },
    ("vat", "b"): {
        "path": "experiments/train_vat_baseline/2026-03-11_19-58-20/epoch_7.pt",
        "sha256": "db09c5605300cd31c5ff7030d8e2f468aafb1e215f556c1a05daf8fb034bff0d",
        "kind": "decoder",
    },
    ("vat", "l"): {
        "path": "experiments/train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt",
        "sha256": "979ddf804cc6b91821e2899c5bb9f1e92dec6a1136f25b26cb3d62a41d25b673",
        "kind": "decoder",
    },
}
