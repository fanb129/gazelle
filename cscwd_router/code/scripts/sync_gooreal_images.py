"""Preserve exactly the official GOO test RGB images for offline use."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[3]
REMOTE = r'''
import hashlib,io,json,sys,tarfile,zipfile
from pathlib import Path
root=Path('/newhome/fb/dataset/gooreal_data')
records=json.loads((root/'gooreal_test_preprocessed.json').read_text())
paths=sorted({x['path'] for x in records})
archive=root/'.gooreal_zip_cache/finalrealdatasetImgsV3.zip'
rows=[]
with zipfile.ZipFile(archive) as source, tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as target:
    for path in paths:
        data=source.read(path)
        item=tarfile.TarInfo(path); item.size=len(data)
        target.addfile(item,io.BytesIO(data))
        rows.append({'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                     'server_archive':str(archive),'archive_member':path})
    data=json.dumps({'images':len(rows),'queries':len(records),'files':rows},indent=2).encode()
    item=tarfile.TarInfo('GOO_RGB_SOURCE_MANIFEST.json'); item.size=len(data)
    target.addfile(item,io.BytesIO(data))
'''


def run(args):
    root = Path(args.local_root).resolve()
    destination = root / "gooreal_data"
    destination.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("*\n!.gitignore\n")
    command = "/home/fb/anaconda3/envs/py310/bin/python -c " + shlex.quote(REMOTE)
    process = subprocess.Popen(["ssh", args.host, command], stdout=subprocess.PIPE)
    actual = {}
    with tarfile.open(fileobj=process.stdout, mode="r|") as source:
        for item in source:
            path = Path(item.name)
            if path.is_absolute() or ".." in path.parts or not item.isfile():
                raise ValueError("unexpected GOO image archive member")
            data = source.extractfile(item).read()
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            actual[item.name] = hashlib.sha256(data).hexdigest()
            if len(actual) % 250 == 0:
                print(json.dumps({"downloaded_images": len(actual)}), flush=True)
    if process.wait() != 0:
        raise RuntimeError("GOO test RGB transfer failed")
    manifest = json.loads((destination / "GOO_RGB_SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        if actual[item["path"]] != item["sha256"]:
            raise ValueError("GOO RGB source/copy SHA256 mismatch: " + item["path"])
        item["local_path"] = str((destination / item["path"]).relative_to(ROOT))
    manifest.update(completed=True, host=args.host, bytes=sum(x["bytes"] for x in manifest["files"]),
                    verification="every transferred RGB matched the source zip member SHA256")
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("completed", "images", "queries", "bytes")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="fb@3090.lab")
    p.add_argument("--local-root", default=str(ROOT / "cscwd_router/results/local_artifacts_20261125"))
    p.add_argument("--manifest", default=str(ROOT / "cscwd_router/results/OFFLINE_GOO_RGB_MANIFEST.json"))
    run(p.parse_args())
