"""Server-side structural smoke test for the routed DINOv3 backbone."""

import argparse

import torch

from gazelle.routing.backbone import RoutedDinoV3Backbone
from gazelle.routing.losses import coverage_router_loss
from gazelle.routing.router import CoverageAwareSpatialRouter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("dinov3_vitb16", "dinov3_vitl16"), default="dinov3_vitb16")
    parser.add_argument("--route_after_block", type=int, default=None)
    parser.add_argument("--keep_ratio", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    route_after_block = args.route_after_block
    if route_after_block is None:
        route_after_block = 5 if "vitb" in args.backbone else 11

    device = torch.device(args.device)
    torch.manual_seed(3106)
    backbone = RoutedDinoV3Backbone(args.backbone).to(device).eval()
    images = torch.randn(args.batch_size, 3, 512, 512, device=device)

    with torch.inference_mode():
        dense_outputs = backbone(images)
        prefix = backbone.forward_prefix(images, route_after_block)
        all_indices = torch.arange(prefix.height * prefix.width, device=device).repeat(args.batch_size, 1)
        routed_all_outputs = backbone.forward_suffix(prefix, all_indices)

        max_errors = []
        mean_errors = []
        for dense, routed in zip(dense_outputs, routed_all_outputs):
            max_errors.append(float((dense - routed).abs().max().item()))
            mean_errors.append(float((dense - routed).abs().mean().item()))
            torch.testing.assert_close(dense, routed, atol=args.atol, rtol=args.rtol)

        route_features = backbone.tokens_to_map(prefix.tokens, prefix.height, prefix.width)
        router = CoverageAwareSpatialRouter(
            in_dim=backbone.get_dimension(),
            hidden_dim=256,
            keep_ratio=args.keep_ratio,
            escape_tokens=8,
        ).to(device).eval()
        bboxes = [[[0.1, 0.1, 0.25, 0.30]] for _ in range(args.batch_size)]
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            routing = router(route_features, bboxes)
            sparse_outputs = backbone.forward_suffix(prefix, routing.image_keep_indices)

        num_patches = prefix.height * prefix.width
        if routing.image_keep_indices.min() < 0 or routing.image_keep_indices.max() >= num_patches:
            raise AssertionError("router returned an out-of-range patch index")
        if routing.image_keep_indices.shape[1] > 1:
            if not (routing.image_keep_indices[:, 1:] > routing.image_keep_indices[:, :-1]).all():
                raise AssertionError("router indices must be sorted and unique")

        targets = torch.zeros(args.batch_size, 64, 64, device=device)
        targets[:, 28:36, 44:52] = 1.0
        losses = coverage_router_loss(
            routing,
            targets,
            inout=torch.ones(args.batch_size, device=device),
        )
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise AssertionError(f"non-finite router loss detected: {losses}")

    if backbone.prefix_token_count != 5:
        raise AssertionError(
            f"expected 1 CLS + 4 storage tokens, got {backbone.prefix_token_count} special tokens"
        )
    print(
        "PASS dense equivalence; "
        f"max absolute errors: {max_errors}; mean absolute errors: {mean_errors}"
    )
    print(
        "PASS sparse shapes; "
        f"special_tokens={backbone.prefix_token_count}, "
        f"requested_ratio={args.keep_ratio:.4f}, actual_ratio={routing.actual_keep_ratio:.4f}, "
        f"kept={routing.image_keep_indices.shape[1]}/{prefix.height * prefix.width}, "
        f"outputs={[tuple(output.shape) for output in sparse_outputs]}"
    )
    loss_values = {key: float(value) for key, value in losses.items()}
    print(f"PASS AMP/FP32 router losses finite: {loss_values}")


if __name__ == "__main__":
    main()
