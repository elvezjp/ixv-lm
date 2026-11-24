"""別プロセスでメモリを見張り、危なくなったら学習を止める。macOS 専用。"""
import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def sysctl(name):
    return subprocess.check_output(["sysctl", "-n", name], text=True).strip()


def swap_used_gb():
    used = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage"))
    return float(used.group(1)) / 1024


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--min-free", type=int, default=5, help="空きメモリの下限（%%）")
    parser.add_argument("--max-swap-gb", type=float, default=6.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--strikes", type=int, default=3, help="連続で超えたら止める回数")
    args = parser.parse_args()

    strikes = 0
    with args.log.open("a", buffering=1) as log:
        while alive(args.pid):
            free = int(sysctl("kern.memorystatus_level"))
            swap = swap_used_gb()
            strikes = strikes + 1 if free < args.min_free or swap > args.max_swap_gb else 0
            log.write(json.dumps({"time": time.time(), "free": free, "swap_gb": swap, "strikes": strikes}) + "\n")
            if strikes >= args.strikes:
                log.write(json.dumps({"time": time.time(), "event": "terminate", "pid": args.pid}) + "\n")
                os.kill(args.pid, signal.SIGTERM)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
