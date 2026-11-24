"""青空文庫の保護期間満了の作品を取得する。相手のサーバーに配慮して 1 秒に 1 件とする。

公式の作品一覧（list_person_all_extended_utf8.csv）で作品著作権フラグが「なし」のものだけを取る。
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path, help="作品一覧の CSV を置いたディレクトリ")
    args = parser.parse_args()

    rows = csv.DictReader((args.dest / "list_person_all_extended_utf8.csv").open(encoding="utf-8-sig"))
    works = {r["作品ID"]: r for r in rows if r["作品著作権フラグ"] == "なし" and r["テキストファイルURL"]}
    out = args.dest / "text"
    out.mkdir(exist_ok=True)
    print(f"{len(works)} 作品", flush=True)
    failed = []
    for n, (work_id, row) in enumerate(works.items(), 1):
        path = out / f"{work_id}.zip"
        if not path.exists():
            part = path.with_suffix(".part")
            done = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-A", "ixv-lm-corpus", "-o", str(part),
                                   row["テキストファイルURL"]], check=False)
            if done.returncode == 0:
                part.rename(path)
            else:
                part.unlink(missing_ok=True)
                failed.append(work_id)
            time.sleep(1)
        if n % 1000 == 0:
            print(f"{datetime.datetime.now():%H:%M:%S} {n}/{len(works)} 失敗 {len(failed)}", flush=True)
    files = [{"id": i, "url": r["テキストファイルURL"], "sha256": hashlib.sha256((out / f"{i}.zip").read_bytes()).hexdigest()}
             for i, r in works.items() if (out / f"{i}.zip").exists()]
    (args.dest / "source.json").write_text(json.dumps(
        {"acquired": f"{datetime.date.today()}", "failed": failed, "files": files}, ensure_ascii=False, indent=1))
    print(f"完了 {len(files)} 作品、失敗 {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
