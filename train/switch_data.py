"""動いている学習を、次の保存の直後に止め、別のデータで最新の保存から続ける。

例: switch_data.py --run ~/ixv-lm-data/runs/v1-0.3b --data ~/ixv-lm-data/processed/v1b
引数は supervisor.jsonl の最後の supervisor_start から引き継ぎ、--data だけ差し替える。
モデルの設定は同じなので、train.py は保存から続けられる（語彙が同じデータに限る）。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3 * 3600, help="保存を待つ上限の秒数")
    args = parser.parse_args()

    events = [json.loads(l) for l in (args.run / "supervisor.jsonl").read_text().splitlines() if l.strip()]
    start = [e for e in events if e["event"] == "supervisor_start"][-1]
    supervisor, train_args = start["pid"], list(start["args"])
    if not alive(supervisor):
        raise SystemExit(f"見張り {supervisor} が動いていません")
    old_vocab = json.loads((Path(train_args[train_args.index("--data") + 1]) / "meta.json").read_text())["vocab_size"]
    new_vocab = json.loads((args.data / "meta.json").read_text())["vocab_size"]
    if old_vocab != new_vocab:
        raise SystemExit("語彙の大きさが違うので続けられません")
    train_args[train_args.index("--data") + 1] = str(args.data)

    latest = (args.run / "latest").read_text()
    print(f"{latest} の次の保存を待つ", flush=True)
    deadline = time.time() + args.timeout
    while (args.run / "latest").read_text() == latest:
        if time.time() > deadline:
            raise SystemExit("保存が来ないので止めずに終えます")
        time.sleep(10)
    saved = (args.run / "latest").read_text()
    print(f"{saved} を保存した。学習を止める", flush=True)

    train = int((args.run / "pid").read_text())
    os.kill(supervisor, signal.SIGTERM)
    while alive(supervisor) or alive(train):
        time.sleep(2)

    out = (args.run / "supervisor.out").open("ab")
    new = subprocess.Popen([sys.executable, str(HERE / "supervise.py"), "--run", str(args.run), "--", *train_args],
                           stdout=out, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
    print(f"{saved} から {args.data} で再開した（見張り {new.pid}）", flush=True)


if __name__ == "__main__":
    main()
