"""T3 CPU structure, backward, data-protocol and serialization checks.

No optimizer updates or gaze dataset evaluations are run on the local machine.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from PIL import Image
import torch
from torch.nn import functional as F
from dinov2.models.vision_transformer import DinoVisionTransformer

from gazelle.routing.losses import coverage_router_loss
from gazelle.utils import gazefollow_auc, gazefollow_l2
from cscwd_router.code.data_prep.gazefollow import GazeBatch, collate_people, make_data, synthetic_batch
from cscwd_router.code.scripts.eval_router_dinov2 import validate_evaluation_request
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone
from cscwd_router.code.stag.checkpoint import clone_state, read_checkpoint, restore_checkpoint, save_checkpoint, state_sha256
from cscwd_router.code.stag.engine import evaluate
from cscwd_router.code.stag.model_dinov2 import build_model
from cscwd_router.code.stag.pipeline import MATCHED_EPOCHS, SPARSE_KEEP_SCHEDULE, SUPPORT_EPOCHS, epoch_seed, initial_equivalence, run_pipeline


def tiny_backbone():
    native = DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24, depth=12,
                                   num_heads=3, mlp_ratio=2, init_values=1.0, block_chunks=0)
    return RoutedDinoV2Backbone(model=native)


class TrainingDinoV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3106)
        torch.set_num_threads(1)
        self.model = build_model({"image_size": 56, "decoder_dim": 32, "decoder_layers": 1,
                                  "router_hidden_dim": 16, "escape_tokens": 0}, backbone=tiny_backbone())
        self.batch = synthetic_batch(2, 56, 3106)

    def test_zero_initialization_is_exactly_the_official_model_at_k100(self):
        checks = initial_equivalence(self.model, self.batch, torch.device("cpu"))
        self.assertTrue(all(value["equal"] and value["finite"] for value in checks.values()))
        self.assertEqual([value["max_abs_diff"] for value in checks.values()], [0.0, 0.0])
        state = clone_state(self.model.decoder)
        self.model.load_official_decoder(state)
        self.assertEqual(set(state), set(self.model.decoder.get_gazelle_state_dict()))

    def test_support_backward_reaches_only_router_and_skips_decoder(self):
        self.model.set_stage("support")
        self.model.train()
        calls = []
        hook = self.model.decoder.register_forward_pre_hook(lambda *_args: calls.append(1))
        routing = self.model.predict_support(self.batch.inputs())
        coverage_router_loss(routing, self.batch.heatmaps)["total"].backward()
        hook.remove()
        self.assertEqual(calls, [])
        self.assertTrue(self.model.encoder.router.training)
        self.assertFalse(self.model.decoder.training)
        self.assertGreater(self.model.encoder.router.score_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.decoder.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.encoder.backbone.parameters()))

    def test_decoder_stages_backward_reaches_only_decoder(self):
        for stage, mode in (("sparse_decoder", "sparse"), ("matched_dense", "dense")):
            with self.subTest(stage=stage):
                self.model.set_stage(stage)
                self.model.encoder.router.keep_ratio = 0.5
                self.model.train()
                output = self.model(self.batch.inputs(), mode=mode)
                F.binary_cross_entropy(torch.cat(output["heatmap"]), self.batch.heatmaps).backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.decoder.parameters() if p.grad is not None), 0)
                self.assertTrue(self.model.decoder.training)
                self.assertFalse(self.model.encoder.router.training)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.encoder.parameters()))
                self.assertFalse(self.model.encoder.backbone.training)

    def test_matched_dense_does_not_read_router_parameters(self):
        self.model.set_stage("matched_dense")
        self.model.eval()
        calls = []
        hook = self.model.encoder.router.register_forward_pre_hook(lambda *_args: calls.append(1))
        with torch.inference_mode():
            before = torch.cat(self.model(self.batch.inputs(), mode="dense")["heatmap"])
            for parameter in self.model.encoder.router.parameters():
                parameter.normal_(mean=2.0, std=1.0)
            after = torch.cat(self.model(self.batch.inputs(), mode="dense")["heatmap"])
        hook.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(before, after))

    def test_checkpoint_roundtrip_preserves_sparse_predictions_without_storing_dino(self):
        self.model.set_stage("sparse_decoder")
        self.model.encoder.router.keep_ratio = 0.5
        self.model.eval()
        context = {"data": {"protocol": "synthetic"}, "external_weights": {"backbone": {"sha256": "fixture"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            save_checkpoint(path, self.model, stage="sparse_decoder", epoch=4, context=context)
            payload = read_checkpoint(path)
        self.assertEqual(payload["checkpoint_role"], "fixed_final")
        self.assertFalse(any(name.startswith("backbone.") for name in payload["decoder"]))
        self.assertNotIn("encoder", payload)
        fresh_backbone = tiny_backbone()
        fresh_backbone.load_state_dict(self.model.encoder.backbone.state_dict(), strict=True)
        fresh = build_model(self.model.config, backbone=fresh_backbone)
        restore_checkpoint(fresh, payload)
        fresh.eval()
        with torch.inference_mode():
            before = torch.cat(self.model(self.batch.inputs())["heatmap"])
            after = torch.cat(fresh(self.batch.inputs())["heatmap"])
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(fresh.encoder.router.keep_ratio, 0.5)
        self.assertEqual(state_sha256(payload["decoder"]), state_sha256(fresh.decoder.state_dict()))

    def test_optimizer_scheduler_initial_state_can_be_restored(self):
        self.model.set_stage("sparse_decoder")
        optimizer = torch.optim.AdamW(self.model.decoder.parameters(), lr=0.0003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.model, stage="sparse_decoder", epoch=-1, context={},
                            optimizer=optimizer, scheduler=scheduler)
            payload = read_checkpoint(path)
        other_optimizer = torch.optim.AdamW(self.model.decoder.parameters(), lr=0.5)
        other_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(other_optimizer, T_max=10)
        restore_checkpoint(self.model, payload, optimizer=other_optimizer, scheduler=other_scheduler)
        self.assertEqual(other_optimizer.param_groups[0]["lr"], 0.0003)
        self.assertEqual(other_scheduler.state_dict(), scheduler.state_dict())

    def test_fixed_recipe_and_matched_epoch_seeds(self):
        self.assertEqual(SUPPORT_EPOCHS, 3)
        self.assertEqual(SPARSE_KEEP_SCHEDULE, (1.0, 0.75, 0.5, 0.5, 0.5))
        self.assertEqual(len(SPARSE_KEEP_SCHEDULE), MATCHED_EPOCHS)
        sparse = [epoch_seed(3106, "sparse_decoder", epoch) for epoch in range(5)]
        matched = [epoch_seed(3106, "matched_dense", epoch) for epoch in range(5)]
        self.assertEqual(sparse, matched)

    def test_official_test_rejects_holdout_support_and_partial_checkpoints(self):
        payload = {"context": {"data": {"protocol": "holdout"}}, "checkpoint_role": "fixed_final", "stage": "sparse_decoder", "epoch": 4}
        with self.assertRaisesRegex(ValueError, "full_train"):
            validate_evaluation_request(payload, "official_test")
        payload["context"]["data"]["protocol"] = "full_train"
        validate_evaluation_request(payload, "official_test")
        for role, stage, epoch in (("epoch_last", "sparse_decoder", 4), ("fixed_final", "support", 2), ("fixed_final", "matched_dense", 2)):
            with self.subTest(role=role, stage=stage, epoch=epoch):
                with self.assertRaisesRegex(ValueError, "fixed-final"):
                    validate_evaluation_request({**payload, "checkpoint_role": role, "stage": stage, "epoch": epoch}, "official_test")

    def test_metrics_match_existing_multi_annotation_protocol_and_person_weighting(self):
        heatmaps = torch.zeros(3, 64, 64)
        for index, (y, x) in enumerate(((10, 20), (30, 40), (50, 5))):
            heatmaps[index, y, x] = 1
        batch = GazeBatch(self.batch.images, self.batch.bboxes, [[0.25, 0.5], [0.2, 0.3], [0.7, 0.8]],
                          [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], torch.ones(3), [80] * 3, [120] * 3)

        class FixedPredictor(torch.nn.Module):
            def forward(self, inputs, *, mode):
                return {"heatmap": [heatmaps[:2], heatmaps[2:]], "routing": None}

        metrics = evaluate(FixedPredictor(), [batch], torch.device("cpu"), mode="dense")
        expected_auc = sum(gazefollow_auc(heatmaps[i], batch.gazex[i], batch.gazey[i], 80, 120) for i in range(3)) / 3
        expected_l2 = [gazefollow_l2(heatmaps[i], batch.gazex[i], batch.gazey[i]) for i in range(3)]
        self.assertAlmostEqual(metrics["auc"], expected_auc)
        self.assertAlmostEqual(metrics["avg_l2"], sum(row[0] for row in expected_l2) / 3)
        self.assertAlmostEqual(metrics["min_l2"], sum(row[1] for row in expected_l2) / 3)
        self.assertEqual((metrics["persons"], metrics["images"]), (3, 2))
        self.assertEqual(metrics["head_count_strata"]["two"]["persons"], 2)
        self.assertEqual(metrics["head_count_strata"]["single"]["persons"], 1)

    def test_pipeline_resume_keeps_completed_epochs_and_official_branch_starts(self):
        # Test orchestration with injected state perturbations, not optimizer updates.
        # All mock histories/checkpoints stay in TemporaryDirectory.
        data = make_data(protocol="synthetic", data_path=None, transform=None, batch_size=2,
                         eval_batch_size=2, workers=0, image_size=56)
        context = {"train_config": {}, "sources": {}, "external_weights": {}}
        calls = []

        def state_update(model, loader, optimizer, device, **kwargs):
            calls.append(model.stage)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(0.0001)
            return {"losses": {"total": 0.5}, "persons": 3, "images": 2, "optimizer_steps": 1}

        def interrupted(*args, **kwargs):
            if len(calls) == 2:
                raise RuntimeError("planned interruption")
            return state_update(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(), patch("builtins.print"):
            warnings.simplefilter("ignore", UserWarning)  # No optimizer runs in this injected orchestration test.
            run_dir, report_dir = Path(directory) / "run", Path(directory) / "reports"
            with patch("cscwd_router.code.stag.pipeline.train_epoch", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                    run_pipeline(self.model, data, torch.device("cpu"), run_dir=run_dir, report_dir=report_dir, context=context)
            self.assertEqual(read_checkpoint(run_dir / "support/last.pt")["epoch"], 1)
            with patch("cscwd_router.code.stag.pipeline.train_epoch", side_effect=state_update):
                result = run_pipeline(self.model, data, torch.device("cpu"), run_dir=run_dir,
                                      report_dir=report_dir, context=context, resume=True)
            self.assertEqual(calls, ["support"] * 3 + ["sparse_decoder"] * 5 + ["matched_dense"] * 5)
            self.assertTrue(result["passed"])
            self.assertEqual([len(stage["history"]) for stage in result["stages"]], [3, 5, 5])
            with patch("cscwd_router.code.stag.pipeline.train_epoch") as train:
                completed = run_pipeline(self.model, data, torch.device("cpu"), run_dir=run_dir,
                                         report_dir=report_dir, context=context, resume=True)
                train.assert_not_called()
            self.assertTrue(completed["passed"])
            self.assertEqual(result["baseline_checkpoint"], completed["baseline_checkpoint"])
            self.assertEqual(len(list(run_dir.rglob("*.pt"))), 7)


class GazeFollowProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        records = []
        for image in range(4):
            Image.new("RGB", (32, 32), color=(image * 20, 80, 100)).save(self.root / f"{image}.png")
            heads = []
            for person in range(2 if image == 0 else 1):
                x, y = 0.6 + person * 0.1, 0.75
                heads.append({"bbox": [2, 2, 12, 12], "bbox_norm": [2/32, 2/32, 12/32, 12/32],
                              "gazex": [x * 32], "gazey": [y * 32], "gazex_norm": [x], "gazey_norm": [y], "inout": 1})
            records.append({"path": f"{image}.png", "heads": heads})
        (self.root / "train_preprocessed.json").write_text(json.dumps(records))

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, protocol):
        return make_data(protocol=protocol, data_path=self.root, transform=lambda image: torch.zeros(3, 56, 56),
                         batch_size=2, eval_batch_size=2, workers=0, val_fraction=0.5, augment=False, image_size=56)

    def test_holdout_is_image_disjoint_and_validation_groups_all_heads(self):
        data = self.make("holdout")
        self.assertTrue(set(data.train_dataset.image_indices).isdisjoint(data.validation_dataset.image_indices))
        self.assertFalse(data.metadata["official_test_accessed"])
        batch = next(iter(data.validation_loader()))
        self.assertEqual(sum(map(len, batch.bboxes)), len(batch.gazex))
        self.assertEqual(batch.images.shape[0], len(batch.bboxes))
        self.assertIsNone(batch.heatmaps)

    def test_full_train_builds_no_evaluation_dataset_and_never_needs_test_json(self):
        data = self.make("full_train")
        self.assertIsNone(data.validation_loader())
        self.assertEqual(data.metadata["train_persons"], 5)
        self.assertEqual(data.metadata["validation_fraction"], 0.0)
        self.assertFalse(data.metadata["official_test_accessed"])
        self.assertFalse((self.root / "test_preprocessed.json").exists())

    def test_person_collation_keeps_heatmaps_and_query_order(self):
        data = self.make("full_train")
        batch = collate_people([data.train_dataset[0], data.train_dataset[1]])
        self.assertEqual(tuple(batch.heatmaps.shape), (2, 64, 64))
        self.assertEqual([len(boxes) for boxes in batch.bboxes], [1, 1])
        self.assertEqual(batch.gazex, [[0.6], [0.7]])

    def test_synthetic_mode_works_without_any_data_path(self):
        data = make_data(protocol="synthetic", data_path=self.root / "does_not_exist", transform=None,
                         batch_size=2, eval_batch_size=2, workers=0, image_size=56)
        self.assertEqual(len(next(iter(data.train_loader(100))).gazex), 3)
        self.assertIs(data.train_loader(100)[0], data.train_loader(200)[0])
        self.assertFalse(data.metadata["official_test_accessed"])


if __name__ == "__main__":
    unittest.main()
