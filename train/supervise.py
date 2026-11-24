"""学習を見張り、GPU のロックアップで止まったときだけ最新の保存から再開する。

例: supervise.py --run ~/ixv-lm-data/runs/v1-0.3b -- --data ~/ixv-lm-data/processed/v1 --steps 200000 ...
`--` の後ろは train.py にそのまま渡す（--run と --resume はこちらで付ける）。
損失の異常、メモリの見張りによる停止、その他のエラーでは再開しない。
シェルのループではなく、このスクリプト1本で見張る（README の「学習マシンでの運用」）。
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
GPU_HANG = "kIOGPUCommandBufferCallbackError"


def other_training():
    """自分以外の train.py が動いていたら、その行を返す。"""
    lines = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
    return [line.strip() for line in lines
            if "train/train.py" in line or line.strip().endswith("train.py") or " train.py " in line]


def last_event(log):
    if not log.exists():
        return None
    lines = log.read_text().strip().splitlines()
    return json.loads(lines[-1]).get("event") if lines else None


def output_since(path, offset):
    """今回の起動で増えた出力だけを返す。前回のロックアップの文言を拾わないため。"""
    with path.open("rb") as f:
        f.seek(offset)
        return f.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--wait", type=int, default=30, help="再開の前に待つ秒数")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = [a for a in args.train_args if a != "--"]

    if running := other_training():
        raise SystemExit(f"別の学習が動いているので始めません: {running}")
    args.run.mkdir(parents=True, exist_ok=True)
    events = (args.run / "supervisor.jsonl").open("a", buffering=1)

    def record(**fields):
        events.write(json.dumps({"time": time.time(), **fields}, ensure_ascii=False) + "\n")

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.setdefault("KDA_METAL", "1")
    stop = {"requested": False}

    def on_signal(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, on_signal)
    record(event="supervisor_start", pid=os.getpid(), args=train_args)

    restarts = 0
    while True:
        resume = (args.run / "latest").exists()
        command = [sys.executable, str(HERE / "train.py"), "--run", str(args.run), *train_args]
        if resume:
            command.append("--resume")
        offset = (args.run / "train.out").stat().st_size if (args.run / "train.out").exists() else 0
        with (args.run / "train.out").open("ab") as out:
            train = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        (args.run / "pid").write_text(str(train.pid))
        caffeinate = subprocess.Popen(["caffeinate", "-is", "-w", str(train.pid)])
        watchdog = subprocess.Popen([sys.executable, str(HERE / "watchdog.py"), "--pid", str(train.pid),
                                     "--log", str(args.run / "watchdog.jsonl"), "--interval", "30"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        record(event="train_start", pid=train.pid, resume=resume, latest=(args.run / "latest").read_text() if resume else None)

        while train.poll() is None:
            if stop["requested"]:
                train.terminate()
                train.wait()
                record(event="stopped_by_signal")
                return
            time.sleep(5)
        watchdog.wait(timeout=60)
        caffeinate.wait(timeout=60)
        code = train.returncode
        record(event="train_exit", code=code)

        if last_event(args.run / "log.jsonl") == "end":
            record(event="finished")
            return
        if GPU_HANG not in output_since(args.run / "train.out", offset):
            record(event="not_restarting", reason="GPU のロックアップ以外で止まった", code=code)
            raise SystemExit(1)
        if restarts >= args.max_restarts:
            record(event="gave_up", restarts=restarts)
            raise SystemExit(2)
        restarts += 1
        record(event="restart", n=restarts, latest=(args.run / "latest").read_text())
        time.sleep(args.wait)


if __name__ == "__main__":
    main()
