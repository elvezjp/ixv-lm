"""リリースを1本で通す。本線から分かれて学習率を下げ、会話の追加学習をし、ツール呼び出しを採点し、本線を再開する。

例: release.py --main ~/ixv-lm-data/runs/v1-0.3b --step 80000 --decay 5000 --name r2 \
      --sft ~/ixv-lm-data/sft/oasst2-ja.jsonl --sft ~/ixv-lm-data/sft/llm-jp-instructions-train.jsonl:5 \
      --sft ~/ixv-lm-data/sft/llm-jp-instructions-dev.jsonl:5 --sft ~/ixv-lm-data/sft/tool-v1.jsonl \
      --tool-eval ~/ixv-lm-data/sft/tool-eval-v1.jsonl --tool-train ~/ixv-lm-data/sft/tool-v1.jsonl
重い処理は1つずつ順に走らせる。途中で失敗したら、そこから先の学習はせず、本線だけは必ず再開する。
進み具合は <main>-<name>.release.jsonl に1行ずつ書く。
"""
import argparse
import json
import os
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


def events(run):
    path = run / "supervisor.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--decay", type=int, required=True)
    parser.add_argument("--name", required=True, help="リリースの名前（r2 など）")
    parser.add_argument("--data", type=Path, default=Path.home() / "ixv-lm-data/processed/v1b")
    parser.add_argument("--sft", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tool-eval", type=Path)
    parser.add_argument("--tool-train", type=Path, action="append", default=[],
                        help="学習に使ったツールの問題（比べる run の分も）。採点から外す")
    parser.add_argument("--compare", type=Path, action="append", default=[],
                        help="同じ問題で採点し直す、前のリリースの追加学習の run")
    parser.add_argument("--wait-hours", type=float, default=48, help="本線が --step に届くのを待つ上限")
    args = parser.parse_args()
    # 子のスクリプトは train/ で動くので、パスは絶対にしておく
    args.main, args.data = args.main.expanduser().resolve(), args.data.expanduser().resolve()
    args.tool_eval = args.tool_eval and args.tool_eval.expanduser().resolve()
    args.tool_train = [p.expanduser().resolve() for p in args.tool_train]
    args.compare = [p.expanduser().resolve() for p in args.compare]
    args.sft = [str(Path(p).expanduser().resolve()) + (":" + r if r else "")
                for p, _, r in (s.partition(":") for s in args.sft)]

    branch = args.main.with_name(f"{args.main.name}-{args.name}")
    sft_run = branch.with_name(branch.name + "-sft")
    log = args.main.with_name(f"{args.main.name}-{args.name}.release.jsonl").open("a", buffering=1)
    main_start = [e for e in events(args.main) if e["event"] == "supervisor_start"][-1]

    def record(**fields):
        line = json.dumps({"time": time.time(), **fields}, ensure_ascii=False)
        log.write(line + "\n")
        print(line, flush=True)

    def run(step, command, env=None):
        record(event="step_start", step=step, command=[str(c) for c in command])
        code = subprocess.run([str(c) for c in command], cwd=HERE,
                              env={**os.environ, "KDA_METAL": "1", **(env or {})}).returncode
        record(event="step_end", step=step, code=code)
        return code == 0

    record(event="release_start", name=args.name, step=args.step, decay=args.decay, branch=str(branch),
           sft_run=str(sft_run))
    stopped_main = False
    try:
        # 1. 本線が ckpt-N を保存したら止めて、分かれた run で学習率を下げ始める
        stopped_main = True  # 失敗しても本線の様子を見て再開する
        if not run("branch", [sys.executable, HERE / "branch_release.py", "--main", args.main, "--step", args.step,
                              "--decay", args.decay, "--out", branch,
                              "--timeout", int(args.wait_hours * 3600)]):
            return
        # 2. 分かれた学習の終わりを待つ
        launched = time.time()
        while True:
            ev = events(branch)
            starts = [e for e in ev if e["event"] == "supervisor_start"]
            if starts and not alive(starts[-1]["pid"]):  # 見張りが終わった（止まり方は下で見る）
                break
            if not starts and time.time() - launched > 600:
                record(event="decay_not_started")
                return
            time.sleep(30)
        last = ev[-1]["event"]
        val = [json.loads(l) for l in (branch / "log.jsonl").read_text().splitlines() if '"eval"' in l]
        record(event="decay_end", result=last, val_loss=val[-1]["val_loss"] if val else None,
               latest=(branch / "latest").read_text())
        if last != "finished":
            return
        # 3. 会話の追加学習（tokens の書式）
        command = [sys.executable, HERE / "sft.py", "--base", branch, "--data", args.data, "--run", sft_run,
                   "--epochs", args.epochs, "--chat-format", "tokens"]
        for spec in args.sft:
            command += ["--sft", spec]
        if not run("sft", command):
            return
        # 4. ツール呼び出しの採点
        if args.tool_eval:
            for target in [sft_run, *args.compare]:
                command = [sys.executable, HERE / "eval_tools.py", "--run", target, "--data", args.data,
                           "--set", args.tool_eval]
                for path in args.tool_train:
                    command += ["--train", path]
                run(f"eval_tools {target.name}", command)
    finally:
        # 5. 本線を再開する（見張りが動いていなければ）
        if stopped_main:
            current = [e for e in events(args.main) if e["event"] == "supervisor_start"][-1]
            if alive(current["pid"]):
                record(event="main_already_running", pid=current["pid"])
            elif events(args.main)[-1]["event"] == "finished":  # 本線が計画の終点まで学習し終えていたら、再開しない
                record(event="main_finished", latest=(args.main / "latest").read_text())
            else:
                out = (args.main / "supervisor.out").open("ab")
                proc = subprocess.Popen([sys.executable, str(HERE / "supervise.py"), "--run", str(args.main), "--",
                                         *main_start["args"]],
                                        stdout=out, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
                record(event="main_resumed", pid=proc.pid, latest=(args.main / "latest").read_text())
        record(event="release_end")


if __name__ == "__main__":
    main()
