"""Recover VAT person tracks from native annotation CSVs (.txt or .csv).

Output is a list of sequences, each containing tracks and chronological frames.
Only the current row supplies a frame's head box: the old preprocessor's centered
11-frame box smoothing is deliberately omitted for causal RGB inference.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def make_track(rows, annotation_path, data_path, sequence_path, width, height, split):
    person_id = annotation_path.stem
    frames = []
    for row_number, row in enumerate(rows, start=1):
        filename, *coordinates = row
        frame_index = int(Path(filename).stem)
        if frames and frame_index <= frames[-1]["frame_index"]:
            raise ValueError(
                f"{annotation_path}:{row_number}: frame indices must increase in CSV order"
            )
        xmin, ymin, xmax, ymax, gazex, gazey = map(float, coordinates)
        # Match the existing VAT gaze-label convention, including mixed signs.
        if gazex >= 0 and gazey < 0:
            gazey = 0.0
        elif gazey >= 0 and gazex < 0:
            gazex = 0.0
        bbox = [max(xmin, 0), max(ymin, 0), min(xmax, width), min(ymax, height)]
        frames.append({
            "path": (sequence_path / filename).as_posix(),
            "frame_index": frame_index,
            "person_id": person_id,
            "bbox_norm": [bbox[0] / width, bbox[1] / height,
                          bbox[2] / width, bbox[3] / height],
            "gazex_norm": [gazex / width],
            "gazey_norm": [gazey / height],
            "inout": int(gazex >= 0 and gazey >= 0),
        })
    return {
        "person_id": person_id,
        "track_id": f"{split}/{sequence_path.relative_to('images').as_posix()}/{person_id}",
        "source_csv": annotation_path.relative_to(data_path).as_posix(),
        "frames": frames,
    }


def preprocess_split(data_path, split):
    sequences = []
    csv_count = csv_rows = 0
    annotation_digest = hashlib.sha256()
    sequence_dirs = sorted(path for path in (data_path / "annotations" / split).glob("*/*")
                           if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError(f"No VAT annotation sequences under {data_path / 'annotations' / split}")
    for sequence_dir in sequence_dirs:
        person_files = sorted(path for path in sequence_dir.iterdir()
                              if path.suffix.lower() in {".txt", ".csv"})
        if not person_files:
            raise FileNotFoundError(f"No person annotation CSVs in {sequence_dir}")
        sequence_path = Path("images") / sequence_dir.parent.name / sequence_dir.name
        first_rows = read_rows(person_files[0])
        with Image.open(data_path / sequence_path / first_rows[0][0]) as image:
            width, height = image.size
        tracks = []
        for index, person_file in enumerate(person_files):
            rows = first_rows if index == 0 else read_rows(person_file)
            tracks.append(make_track(rows, person_file, data_path, sequence_path, width, height, split))
            csv_count += 1
            csv_rows += len(rows)
            # A compact manifest fingerprint covers every native person CSV.
            annotation_digest.update(person_file.relative_to(data_path).as_posix().encode())
            annotation_digest.update(b"\0")
            annotation_digest.update(bytes.fromhex(sha256(person_file)))
        sequences.append({
            "path": sequence_path.as_posix(), "width": width, "height": height,
            "tracks": tracks,
        })
        print(f"{split}: {sequence_path}: {len(tracks)} tracks", flush=True)
    return sequences, {
        "person_csv_files": csv_count,
        "csv_rows": csv_rows,
        "annotation_manifest_sha256": annotation_digest.hexdigest(),
    }


def summarize(sequences, source, legacy_path):
    frame_counts = Counter()
    inout_counts = Counter()
    track_count = gap_count = 0
    for sequence in sequences:
        track_count += len(sequence["tracks"])
        for track in sequence["tracks"]:
            frames = track["frames"]
            frame_counts.update(frame["path"] for frame in frames)
            inout_counts.update(frame["inout"] for frame in frames)
            gap_count += sum(current["frame_index"] - previous["frame_index"] > 1
                             for previous, current in zip(frames, frames[1:]))

    with legacy_path.open() as stream:
        legacy = json.load(stream)
    legacy_counts = Counter()
    for sequence in legacy:
        for frame in sequence["frames"]:
            legacy_counts[frame["path"]] += len(frame["heads"])
    person_frames = sum(frame_counts.values())
    checks = {
        "tracks_match_person_csv_files": track_count == source["person_csv_files"],
        "person_frames_match_csv_rows": person_frames == source["csv_rows"],
        "sequence_paths_match_legacy": {seq["path"] for seq in sequences} == {seq["path"] for seq in legacy},
        "unique_frames_match_legacy": len(frame_counts) == len(legacy_counts),
        "person_frames_match_legacy": person_frames == sum(legacy_counts.values()),
        "per_frame_head_counts_match_legacy": frame_counts == legacy_counts,
    }
    return {
        **source,
        "sequences": len(sequences), "tracks": track_count,
        "unique_image_frames": len(frame_counts), "person_frames": person_frames,
        "in_frame_person_frames": inout_counts[1], "out_of_frame_person_frames": inout_counts[0],
        "frame_index_gaps": gap_count,
        "legacy": {
            "path": str(legacy_path), "sha256": sha256(legacy_path),
            "sequences": len(legacy), "unique_image_frames": len(legacy_counts),
            "person_frames": sum(legacy_counts.values()),
        },
        "checks": checks, "passed": all(checks.values()),
    }


def record_provenance(path, report, report_path):
    lines = [
        "", "## T1 track 恢复（服务器运行）", "",
        f"- UTC：`{report['created_at_utc']}`；git commit：`{report['git_commit']}`。",
        f"- 脚本 SHA256：`{report['script_sha256']}`。",
        f"- 命令：`{report['command']}`。",
        f"- 小结果：`{report_path}`；框输入：当前帧原始真实框，未作时序平滑。",
    ]
    for split, result in report["splits"].items():
        lines.extend([
            f"- {split}：{result['tracks']} tracks / {result['unique_image_frames']} 图像帧 / "
            f"{result['person_frames']} 人物帧；与原 CSV 和拍平 JSON 核对通过。",
            f"  - track JSON（服务器保留）：`{result['output']['path']}`；SHA256：`{result['output']['sha256']}`。",
            f"  - 原始标注 manifest SHA256：`{result['annotation_manifest_sha256']}`；"
            f"原拍平 JSON SHA256：`{result['legacy']['sha256']}`。",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Server-only directory for the full track JSONs")
    parser.add_argument("--legacy-dir", type=Path, required=True,
                        help="Directory containing the original {split}_preprocessed.json")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "cscwd/results/t1_track_recovery.json")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md")
    args = parser.parse_args()
    report = {
        "task": "T1", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "command": shlex.join([sys.executable, *sys.argv]),
        "data_path": str(args.data_path.resolve()),
        "protocol": {
            "bbox": "current_frame_raw_ground_truth_clipped_to_image",
            "temporal_smoothing": "none", "frame_order": "native_csv_order",
            "person_id": "native_annotation_filename_stem_local_to_sequence",
            "gaze": "legacy_mixed_sign_repair_then_normalize; singleton_lists",
            "gaps": "preserved_in_frame_index; no_interpolation",
        },
        "splits": {},
    }
    for split in args.splits:
        sequences, source = preprocess_split(args.data_path.resolve(), split)
        output_path = (args.output_dir / f"{split}_preprocessed_tracks.json").resolve()
        write_json(output_path, sequences)
        result = summarize(sequences, source, (args.legacy_dir / f"{split}_preprocessed.json").resolve())
        result["output"] = {"path": str(output_path), "sha256": sha256(output_path)}
        report["splits"][split] = result
        print(f"{split}: tracks={result['tracks']} image_frames={result['unique_image_frames']} "
              f"person_frames={result['person_frames']} passed={result['passed']}", flush=True)
    report["passed"] = all(result["passed"] for result in report["splits"].values())
    write_json(args.report, report)
    if not report["passed"]:
        raise SystemExit(f"Track count verification failed; see {args.report}")
    record_provenance(args.provenance, report, args.report.resolve())
    print(f"T1 verification passed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
