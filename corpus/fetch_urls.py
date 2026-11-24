"""URL の一覧からファイルを取得する。サイズを確かめ、途中から再開できる。

例: fetch_urls.py urls.txt ~/ixv-lm-data/raw/hplt3-ja
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def remote_size(url):
    head = subprocess.run(["curl", "-sI", "-A", "ixv-lm-corpus", url], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"content-length: (\d+)", head, re.I).group(1))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, dest, limit_rate):
    final = dest / url.rsplit("/", 1)[1]
    size = remote_size(url)
    if not final.exists():
        part = final.with_name(final.name + ".part")
        for attempt in range(5):
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-", "-A", "ixv-lm-corpus",
                            "--limit-rate", limit_rate, "-o", str(part), url], check=False)
            if part.exists() and part.stat().st_size == size:
                part.rename(final)
                break
        else:
            raise RuntimeError(f"取得できません: {url}")
    if final.stat().st_size != size:
        raise RuntimeError(f"サイズが違います: {final}")
    return {"url": url, "name": final.name, "size": size, "sha256": sha256(final)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-rate", default="25M", help="1本あたりの上限。回線を圧迫しないため")
    args = parser.parse_args()

    urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip()]
    args.dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        files = list(pool.map(lambda u: fetch(u, args.dest, args.limit_rate), urls))
    (args.dest / "source.json").write_text(json.dumps(
        {"acquired": f"{datetime.date.today()}", "files": files}, ensure_ascii=False, indent=1))
    print("完了", flush=True)


if __name__ == "__main__":
    main()
