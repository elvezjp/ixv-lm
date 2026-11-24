"""国会会議録検索システムの API から全会議録を取得する。公式の求めどおり、1 本ずつ間隔をあけて取る。

期間を固定し、10 件ごとの応答を pages/ に gzip で保存する。保存済みの分は飛ばすので、途中から再開できる。
"""
import argparse
import datetime
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://kokkai.ndl.go.jp/api/meeting"
PER_PAGE = 10  # API の上限


def get(start, since, until):
    query = urllib.parse.urlencode({"from": since, "until": until, "startRecord": start,
                                    "maximumRecords": PER_PAGE, "recordPacking": "json"})
    request = urllib.request.Request(f"{URL}?{query}", headers={"User-Agent": "ixv-lm-corpus"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            if "meetingRecord" in (data := json.loads(body)):
                return body, data
            print(f"応答に本文がありません start={start}: {body[:200]!r}", flush=True)
        except OSError as error:
            print(f"再試行 start={start}: {error}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"取得できません: start={start}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path)
    parser.add_argument("--from", dest="since", default="1947-01-01")
    parser.add_argument("--until", default="2026-11-06")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    pages = args.dest / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    start, total = 1, None
    while total is None or start <= total:
        path = pages / f"{start:07d}.json.gz"
        if path.exists() and total is not None:
            start += PER_PAGE
            continue
        body, data = get(start, args.since, args.until)
        total = data["numberOfRecords"]
        path.write_bytes(gzip.compress(body))
        if (start // PER_PAGE) % 500 == 0:
            print(f"{datetime.datetime.now():%H:%M:%S} {start}/{total}", flush=True)
        start += PER_PAGE
        time.sleep(args.interval)
    (args.dest / "source.json").write_text(json.dumps({
        "url": URL, "from": args.since, "until": args.until, "records": total,
        "acquired": f"{datetime.date.today()}"}, ensure_ascii=False, indent=1))
    print(f"完了 {total} 件", flush=True)


if __name__ == "__main__":
    main()
