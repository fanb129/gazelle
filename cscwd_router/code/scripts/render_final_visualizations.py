"""Render exported real predictions on CPU; no dataset, checkpoint or GPU required."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


COLORS = ("#e64b35", "#4dbbd5", "#00a087")


def annotate(ax, heads, predictions=None, person=None, ground_truth=True, lines=False):
    indices = range(len(heads)) if person is None else [person]
    for index in indices:
        head = heads[index]
        color = COLORS[index % len(COLORS)]
        x0, y0, x1, y1 = head["bbox_norm"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.4))
        ax.text(x0, y0, f"P{index + 1}", color="white", fontsize=8, va="bottom",
                bbox={"facecolor": color, "alpha": .85, "pad": 1, "edgecolor": "none"})
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if ground_truth and head["inout"]:
            ax.scatter(head["gazex_norm"], head["gazey_norm"], marker="*", s=65, c=color, edgecolors="black", linewidths=.4)
            if lines:
                for x, y in zip(head["gazex_norm"], head["gazey_norm"]):
                    ax.plot([center[0], x], [center[1], y], color=color, linewidth=1, alpha=.9)
        if predictions is not None:
            x, y = predictions["prediction_xy"][index]
            ax.scatter([x], [y], marker="x", s=45, c="white", linewidths=1.5)
            if lines:
                ax.plot([center[0], x], [center[1], y], color="white", linewidth=1, linestyle="--")


def background(ax, image):
    ax.imshow(image, extent=(0, 1, 1, 0), aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def heat_panel(ax, image, case, prediction, person, title, vmax):
    background(ax, image)
    ax.imshow(np.asarray(prediction["heatmaps"][person]), extent=(0, 1, 1, 0),
              vmin=0, vmax=vmax, cmap="turbo", alpha=.62, aspect="auto")
    annotate(ax, case["heads"], prediction, person, lines=True)
    error = prediction["errors"][person]
    suffix = f"L2={next(iter(error.values())):.3f}" if error else "GT out-of-frame"
    if prediction.get("inout_scores") is not None:
        suffix += f"  p(in)={prediction['inout_scores'][person]:.2f}"
    ax.set_title(title + "\n" + suffix, fontsize=9)


def support_panel(ax, image, case, person, vmax):
    prediction = case["predictions"]["union"]
    background(ax, image)
    ax.imshow(np.asarray(prediction["support_probs"][person]), extent=(0, 1, 1, 0),
              vmin=0, vmax=vmax, cmap="magma", alpha=.7, aspect="auto")
    annotate(ax, case["heads"], person=person)
    ax.set_title(f"P{person + 1} conditioned support", fontsize=9)


def token_panel(ax, image, heads, mask, person=None):
    """Show the actual discrete execution mask, with dropped patches dimmed."""
    mask = np.asarray(mask, dtype=np.uint8)
    pixels = np.asarray(Image.fromarray(mask * 255).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)) > 0
    selected = np.clip(image.astype(float) * np.where(pixels[..., None], 1., .13), 0, 255).astype(np.uint8)
    background(ax, selected)
    annotate(ax, heads, person=person)
    ax.set_title(f"{int(mask.sum())}/{mask.size} patches in suffix ({mask.mean():.0%})", fontsize=9)


def row_label(ax, label):
    ax.text(-.07, .5, label, transform=ax.transAxes, ha="right", va="center", fontsize=10, fontweight="bold")


def save(fig, stem, paths):
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    paths.extend([stem.with_suffix(".png"), stem.with_suffix(".svg")])
    plt.close(fig)


def render(manifest_path, output_dir):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    cases = []
    for item in manifest["cases"]:
        path = manifest_path.parent / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"visualization case SHA256 mismatch: {path}")
        case = json.loads(path.read_text())
        image_path = manifest_path.parent / case["image_relative_path"]
        expected = manifest["artifacts"][f"image_{case['id']}"]["sha256"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"visualization RGB SHA256 mismatch: {image_path}")
        with Image.open(image_path) as image:
            image = np.asarray(image.convert("RGB")).copy()
        cases.append((case, image))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset in ("gazefollow", "vat"):
        singles = [(c, image) for c, image in cases if c["kind"] == "single" and c["dataset"] == dataset]
        if not singles:
            continue
        fig, axes = plt.subplots(6, len(singles), figsize=(3.1 * len(singles) + 1.1, 12), squeeze=False, constrained_layout=True)
        for column, (case, image) in enumerate(singles):
            background(axes[0, column], image)
            annotate(axes[0, column], case["heads"], ground_truth=False)
            axes[0, column].set_title(f"{case['id']} ({case['error_bucket']})", fontsize=9)
            background(axes[1, column], image)
            annotate(axes[1, column], case["heads"], lines=True)
            support_panel(axes[2, column], image, case, 0, max(np.max(case["predictions"]["union"]["support_probs"]), 1e-12))
            token_panel(axes[3, column], image, case["heads"], case["predictions"]["union"]["kept_masks"][0])
            vmax = max(np.max(case["predictions"][key]["heatmaps"]) for key in ("official", "union"))
            heat_panel(axes[4, column], image, case, case["predictions"]["official"], 0, "Official GazeLLE", vmax)
            heat_panel(axes[5, column], image, case, case["predictions"]["union"], 0, "Router K50", vmax)
        for row, label in enumerate(("RGB + query", "Ground truth", "Conditioned\nsupport", "Executed\npatches", "GazeLLE\nheatmap", "Router\nheatmap")):
            row_label(axes[row, 0], label)
        fig.suptitle(f"{dataset}: real support -> retained patches -> gaze prediction\nGT star / predicted cross; heatmap scale shared per case", fontsize=11)
        save(fig, output_dir / f"{dataset}_comparison", paths)
    for case, image in cases:
        if case["kind"] != "multi":
            continue
        n = len(case["heads"])
        fig, axes = plt.subplots(3, n + 1, figsize=(3 * (n + 1), 8.5), squeeze=False, constrained_layout=True)
        background(axes[0, 0], image)
        annotate(axes[0, 0], case["heads"])
        axes[0, 0].set_title(f"{case['id']}: {n} GT queries", fontsize=9)
        support_max = max(np.max(case["predictions"]["union"]["support_probs"]), 1e-12)
        for person in range(n):
            support_panel(axes[0, person + 1], image, case, person, support_max)
        union = np.asarray(case["predictions"]["union"]["kept_masks"][0])
        separate = np.asarray(case["predictions"]["independent"]["kept_masks"]).sum(0)
        for row, mask, title in ((1, union, f"Shared union: {int(union.sum())} patches"),
                                 (2, separate, f"Independent: {int(separate.sum())} patch evaluations")):
            background(axes[row, 0], image)
            axes[row, 0].imshow(mask, extent=(0, 1, 1, 0), cmap="viridis", alpha=.65, vmin=0, vmax=max(1, mask.max()), aspect="auto")
            annotate(axes[row, 0], case["heads"])
            axes[row, 0].set_title(title, fontsize=9)
        for person in range(n):
            vmax = max(np.max(case["predictions"][key]["heatmaps"][person]) for key in ("union", "independent"))
            for row, key in ((1, "union"), (2, "independent")):
                heat_panel(axes[row, person + 1], image, case, case["predictions"][key], person, f"{key} P{person + 1}", vmax)
        fig.suptitle("Same RGB and GT queries; fixed K50 per route; GT star / predicted cross", fontsize=11)
        save(fig, output_dir / case["id"], paths)
        # Contribution view: the same scene with only the query person changed.
        fig, axes = plt.subplots(4, n + 1, figsize=(3 * (n + 1) + .8, 10.5), squeeze=False, constrained_layout=True)
        for person in range(n):
            background(axes[0, person], image)
            annotate(axes[0, person], case["heads"], person=person, lines=True)
            axes[0, person].set_title(f"Same RGB, query P{person + 1}", fontsize=9)
            support_panel(axes[1, person], image, case, person, support_max)
            independent = case["predictions"]["independent"]
            token_panel(axes[2, person], image, case["heads"], independent["kept_masks"][independent["person_to_image"][person]], person)
            heat_panel(axes[3, person], image, case, case["predictions"]["union"], person, f"Shared-route output P{person + 1}",
                       max(np.max(case["predictions"]["union"]["heatmaps"]), 1e-12))
        background(axes[0, n], image)
        annotate(axes[0, n], case["heads"])
        axes[0, n].set_title(f"All {n} queries together", fontsize=9)
        background(axes[1, n], image)
        union_probs = np.asarray(case["predictions"]["union"]["union_probs"][0])
        axes[1, n].imshow(union_probs, extent=(0, 1, 1, 0), cmap="magma", alpha=.7, vmin=0, vmax=support_max, aspect="auto")
        annotate(axes[1, n], case["heads"])
        axes[1, n].set_title("Pointwise max -> shared support", fontsize=9)
        token_panel(axes[2, n], image, case["heads"], union)
        axes[3, n].axis("off")
        axes[3, n].text(.05, .8, f"ONE dense prefix\nMAX support + priority -> Top-K\nONE sparse suffix ({int(union.sum())} patches)\n{n} query outputs\n\nIndependent: {int(separate.sum())} patch evaluations\nShared: {int(union.sum())} patch evaluations\n(suffix patch slots; special tokens excluded)",
                        transform=axes[3, n].transAxes, va="top", fontsize=9, linespacing=1.6)
        for row, label in enumerate(("Query + GT", "Learned\nsupport", "Discrete\nexecution", "Final gaze")):
            row_label(axes[row, 0], label)
        fig.suptitle("Person-conditioned allocation and a fixed shared budget\nIndividual masks are separate K50 routes; shared mask is Top-K after max-union and head/escape priority", fontsize=11)
        save(fig, output_dir / (case["id"] + "_mechanism"), paths)
    report = {"source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_forward_executed": False, "figures": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RENDER COMPLETE {output_dir} ({len(paths)} files)", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.output_dir)
