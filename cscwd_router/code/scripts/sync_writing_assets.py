"""Mirror selected paper artifacts locally, with remote and local SHA256 checks."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = "/home/fb/src/paper/gazelleV1"
GROUPS = (
    "testselected_k050_seed0/gazefollow", "vat_from_gfb_router_k050_joint_seed0",
    "vat_from_gfl_router_k050_joint_seed0", "final_k100_seed0", "final_ablation_seed0",
    "final_multi_seed0", "self_multi_timing_seed0", "final_visualizations_seed0",
    "final_visualizations_inframe3_seed0", "gooreal_zero_shot_seed0",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def references(value, key=""):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield key, value
        for name, child in value.items():
            yield from references(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from references(child, key)


def make_plan(include_backbones):
    selected = {}
    local = {}
    result_root = ROOT / "cscwd_router/results"
    for group in GROUPS:
        for path in (result_root / group).rglob("*.json"):
            local[str(path.relative_to(ROOT))] = sha256(path)
            for key, ref in references(json.loads(path.read_text())):
                remote = ref["path"]
                suffix = Path(remote).suffix
                wanted = (remote.endswith(".jsonl.gz") or key == "source_image"
                          or (suffix == ".pt" and "resume" not in key)
                          or (include_backbones and suffix == ".pth"))
                if not wanted:
                    continue
                # Identical native weights also appear under evaluation worktrees.
                if "/checkpoints/" in remote:
                    remote = SERVER_ROOT + "/checkpoints/" + Path(remote).name
                item = {"path": remote, "expected_sha256": ref["sha256"]}
                if remote in selected and selected[remote] != item:
                    raise ValueError(f"conflicting source hashes: {remote}")
                selected[remote] = item
    for path in (ROOT / "cscwd_router/latex/figures").rglob("*"):
        if path.is_file():
            local[str(path.relative_to(ROOT))] = sha256(path)
    extra = [f"/newhome/fb/dataset/{dataset}/{split}_preprocessed.json"
             for dataset in ("gazefollow_extended", "videoattentiontarget")
             for split in ("train", "test")]
    extra += ["/newhome/fb/dataset/gooreal_data/" + name for name in (
        "gooreal_test_preprocessed.json", "testrealhumansNew.pickle")]
    extra += [SERVER_ROOT + "/checkpoints/gazelle_dinov2_vit" + size + "14" + suffix + ".pt"
              for size in ("b", "l") for suffix in ("", "_inout")]
    if include_backbones:
        extra += [SERVER_ROOT + f"/checkpoints/dinov2_vit{size}14_pretrain.pth" for size in ("b", "l")]
    for path in extra:
        selected.setdefault(path, {"path": path, "expected_sha256": None})
    return list(selected.values()), local


REMOTE_INVENTORY = '''
import hashlib,json,os,sys
items=json.load(sys.stdin)
for item in items:
    h=hashlib.sha256()
    with open(item['path'],'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    item['sha256']=h.hexdigest()
    item['bytes']=os.path.getsize(item['path'])
    if item['expected_sha256'] and item['expected_sha256']!=item['sha256']:
        raise ValueError('remote source hash mismatch: '+item['path'])
print(json.dumps(items))
'''


def run(args):
    target = Path(args.local_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # The archive itself must never be staged, including image and JSON files.
    (target / ".gitignore").write_text("*\n!.gitignore\n")
    plan, local = make_plan(args.include_backbones)
    command = args.remote_python + " -c " + shlex.quote(REMOTE_INVENTORY)
    inventory = json.loads(subprocess.check_output(
        ["ssh", args.host, command], input=json.dumps(plan).encode()))
    pending = []
    for item in inventory:
        destination = target / "server" / item["path"].lstrip("/")
        item["local_path"] = str(destination.relative_to(ROOT))
        if not destination.exists() or sha256(destination) != item["sha256"]:
            pending.append(item["path"].lstrip("/"))
    print(json.dumps({"files": len(inventory), "transfer_files": len(pending),
                      "bytes": sum(x["bytes"] for x in inventory)}), flush=True)
    if pending:
        listing = target / "transfer_files.txt"
        listing.write_text("\n".join(pending) + "\n")
        (target / "server").mkdir(exist_ok=True)
        subprocess.run(["rsync", "-a", "--partial", "--progress", "--files-from=" + str(listing),
                        args.host + ":/", str(target / "server") + "/"], check=True)
    for item in inventory:
        if sha256(ROOT / item["local_path"]) != item["sha256"]:
            raise ValueError("local copy hash mismatch: " + item["path"])
    output = {"completed": True, "host": args.host, "local_root": str(target),
              "remote_artifacts": inventory, "local_writing_files": local,
              "files": len(inventory), "bytes": sum(x["bytes"] for x in inventory),
              "verification": "every remote source and local copy SHA256 matched"}
    Path(args.manifest).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("completed", "files", "bytes", "verification")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="fb@3090.lab")
    parser.add_argument("--remote-python", default="/home/fb/anaconda3/envs/py310/bin/python")
    parser.add_argument("--local-root", default=str(ROOT / "cscwd_router/results/local_artifacts_20261125"))
    parser.add_argument("--manifest", default=str(ROOT / "cscwd_router/results/OFFLINE_WRITING_MANIFEST.json"))
    parser.add_argument("--include-backbones", action="store_true")
    run(parser.parse_args())
