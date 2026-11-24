"""本線が ckpt-N を保存したら本線を止め、そこから分かれて学習率を 0 まで下げる run を始める。

例: branch_release.py --main ~/ixv-lm-data/runs/v1-0.3b --step 43000 --decay 5000 --out ~/ixv-lm-data/runs/v1-0.3b-r1
本線の引数は supervisor.jsonl の最後の supervisor_start から引き継ぎ、--steps と --decay-frac だけ替える。
保存は APFS の複製（cp -c）で写すので、本線の保存には触らない。本線は止めたままにする（終わったら手で再開する）。
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


def replace(args, name, value):
    """引数を置き換える。なければ末尾に足す（elvezjp/ixv-lm#9）。"""
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args += [name, str(value)]


def branch_args(train_args, step, decay):
    """分岐した run の引数。本線を止める前に組み立てて、ここで落ちても本線は動いたままにする。"""
    out = list(train_args)
    total = step + decay
    replace(out, "--steps", total)
    replace(out, "--decay-frac", decay / total)
    replace(out, "--keep-every", 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--decay", type=int, required=True, help="学習率を下げるステップ数")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3 * 3600)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} はもうあります")

    events = [json.loads(l) for l in (args.main / "supervisor.jsonl").read_text().splitlines() if l.strip()]
    start = [e for e in events if e["event"] == "supervisor_start"][-1]
    supervisor, train_args = start["pid"], list(start["args"])
    if not alive(supervisor):
        raise SystemExit(f"見張り {supervisor} が動いていません")

    ckpt = f"ckpt-{args.step}"
    print(f"{ckpt} の保存を待つ", flush=True)
    deadline = time.time() + args.timeout
    while (args.main / "latest").read_text() != ckpt:
        if time.time() > deadline:
            raise SystemExit("保存が来ないので止めずに終えます")
        time.sleep(10)
    # 待つ間に見張りが入れ替わっていることがある（再起動のあとの手での再開など）ので、読み直す
    events = [json.loads(l) for l in (args.main / "supervisor.jsonl").read_text().splitlines() if l.strip()]
    start = [e for e in events if e["event"] == "supervisor_start"][-1]
    supervisor, train_args = start["pid"], list(start["args"])
    train = int((args.main / "pid").read_text())
    new_args = branch_args(train_args, args.step, args.decay)
    try:
        os.kill(supervisor, signal.SIGTERM)
    except ProcessLookupError:  # 本線の終点（--steps）で分かれるときは、見張りが自分で終わっていることがある
        pass
    while alive(supervisor) or alive(train):
        time.sleep(2)
    print(f"{ckpt} で本線を止めた", flush=True)

    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.mkdir(parents=True)
    subprocess.run(["cp", "-c", "-R", str(args.main / ckpt), str(tmp / ckpt)], check=True)
    subprocess.run(["cp", str(args.main / "config.json"), str(tmp / "config.json")], check=True)
    (tmp / "latest").write_text(ckpt)
    (tmp / "branch.json").write_text(json.dumps(
        {"main": str(args.main), "step": args.step, "decay": args.decay}, indent=2))
    tmp.rename(args.out)

    total = args.step + args.decay
    out = (args.out / "supervisor.out").open("ab")
    new = subprocess.Popen([sys.executable, str(HERE / "supervise.py"), "--run", str(args.out), "--", *new_args],
                           stdout=out, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
    print(f"{args.out} で {args.step} から {total} まで学習率を下げる（見張り {new.pid}）", flush=True)


if __name__ == "__main__":
    main()
