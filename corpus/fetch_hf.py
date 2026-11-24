"""Hugging Face のデータセットを、版を固定して取得する。sha256 を確かめ、途中から再開できる。

例: fetch_hf.py hotchpotch/fineweb-2-edu-japanese <revision> data/ ~/ixv-lm-data/raw/fineweb-2-edu-japanese
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def listing(repo, revision):
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "ixv-lm-corpus"})
    return json.load(urllib.request.urlopen(request))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def fetch(item, repo, revision, dest, limit_rate):
    name = item["rfilename"]
    final = dest / name
    expected = item["lfs"]["sha256"] if item.get("lfs") else None
    if final.exists() and final.stat().st_size == item["size"]:
        return name, "skip"
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
    for attempt in range(5):
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-A", "ixv-lm-corpus",
                        "--limit-rate", limit_rate, "-o", str(part), url], check=False)
        if part.exists() and part.stat().st_size == item["size"]:
            if expected is None or sha256(part) == expected:
                part.rename(final)
                return name, "ok"
            part.unlink()
    raise RuntimeError(f"取得できません: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("prefix", help="取得するパスの先頭。README.md は常に取得する")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-rate", default="25M", help="1本あたりの上限。回線を圧迫しないため")
    args = parser.parse_args()

    info = listing(args.repo, args.revision)
    items = [s for s in info["siblings"] if s["rfilename"].startswith(args.prefix) or s["rfilename"] == "README.md"]
    args.dest.mkdir(parents=True, exist_ok=True)
    total = sum(s["size"] for s in items)
    print(f"{len(items)} ファイル、{total / 1e9:.1f}GB", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch, s, args.repo, args.revision, args.dest, args.limit_rate) for s in items]
        for job in jobs:
            job.result()
            done += 1
            if done % 50 == 0:
                print(f"{datetime.datetime.now():%H:%M:%S} {done}/{len(items)}", flush=True)
    (args.dest / "source.json").write_text(json.dumps({
        "dataset": args.repo,
        "revision": args.revision,
        "prefix": args.prefix,
        "acquired": f"{datetime.date.today()}",
        "files": [{"name": s["rfilename"], "size": s["size"], "sha256": (s.get("lfs") or {}).get("sha256")} for s in items],
    }, ensure_ascii=False, indent=1))
    print("完了", flush=True)


if __name__ == "__main__":
    main()
