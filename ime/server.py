"""かな漢字変換のサーバー。この Mac の中だけで動かし（127.0.0.1）、入力メソッド（ime/macos）から呼ばれる。

例: .venv/bin/python ime/server.py --dict ~/ixv-lm-data/ime/dict-v1 --run ~/ixv-lm-data/runs/kanji-char-0.1b \
      --data ~/ixv-lm-data/processed/kanji-char-v1
POST /convert {"context": "左の文脈", "kana": "かな", "segments": [3, 4]（省略可）}
  → {"segments": [{"kana": "…", "candidates": ["…", …]}, …], "ms": 123}
  "fast": true ならモデルを使わない（ライブ変換用）
POST /predict {"kana": "かな", "n": 6} → {"candidates": [...]}（読みがかなで始まる語。予測変換）
POST /learn {"kana": "かな", "surface": "選んだ変換"} → {"ok": true}
POST /forget → 学習を全部消す。GET /status → 重み・辞書・使っている計算機（cpu / gpu）・置き場所
/convert と /predict に "learning": false を付けると、学習した変換を使わない（設定画面の「学習を使う」）。
--device auto（既定）は、ふだんは GPU、train.py が動いている間は CPU で計算する（学習と GPU を重ねない）。
中身は ime/engine.py（辞書の下書き → モデルの検算 → 文節ごとの候補）。
MLX は読み込んだスレッドでしか動かないので、1 本のスレッドで順に受ける。
そのため、受けた接続には短いタイムアウトと本文の大きさの上限を付け、1 件で全体が止まらないようにする（elvezjp/ixv-lm#7）。
Web ページから呼ばれないよう、Host が 127.0.0.1 か localhost でないもの、Origin が付いたもの、
POST で Content-Type が application/json でないものは断る（elvezjp/ixv-lm#4）。
"""
import argparse
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ckpt", help="既定は latest")
    parser.add_argument("--store", type=Path, default=Path.home() / "Library/Application Support/IXVKanji",
                        help="学習とユーザー辞書の置き場所")
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--gpu", action="store_true", help="--device gpu と同じ（前の書き方）")
    args = parser.parse_args()
    if args.gpu:
        args.device = "gpu"
    device = {"name": "cpu", "checked": 0.0}

    def pick_device():
        """auto のとき、30 秒ごとに train.py が動いているかを見て、計算機を選ぶ。"""
        if args.device != "auto":
            name = args.device
        elif time.time() - device["checked"] < 30:
            return device["name"]
        else:
            busy = subprocess.run(["pgrep", "-f", "train/train.py"], capture_output=True, text=True).stdout.strip()
            name = "cpu" if busy else "gpu"
            device["checked"] = time.time()
        mx.set_default_device(mx.cpu if name == "cpu" else mx.gpu)
        device["name"] = name
        return name

    pick_device()
    engine = Engine(args.dict, args.run, args.data, args.store, args.ckpt)

    home = str(Path.home())

    def short(path):  # ホームのパス（ユーザー名）は返さない
        return str(path).replace(home, "~", 1)

    def status():
        return {"ckpt": engine.ckpt, "run": short(args.run), "dict": short(args.dict), "device": device["name"],
                "device_setting": args.device, "store": short(args.store), "port": args.port}

    hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
    max_body = 64 * 1024

    class Handler(BaseHTTPRequestHandler):
        timeout = 1  # 本文を送り切らない接続で、ほかの要求を待たせない

        def allowed(self, post):
            """この Mac の入力メソッドからの要求か。ブラウザからの要求（CSRF、DNS rebinding）は断る。"""
            if self.headers.get("Host") not in hosts or self.headers.get("Origin") is not None:
                self.reply({"error": "forbidden"}, 403)
                return False
            if post and not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self.reply({"error": "Content-Type must be application/json"}, 415)
                return False
            return True

        def reply(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.allowed(post=True):
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > max_body:
                    self.reply({"error": "too large"}, 413)
                    return
                body = json.loads(self.rfile.read(length)) if length else {}
                start = time.time()
                pick_device()
                engine.store.enabled = bool(body.get("learning", True))
                engine.store.reload_history()
                if self.path == "/convert":
                    segs = engine.convert(body.get("context", ""), body["kana"], body.get("segments"),
                                          bool(body.get("fast")))
                    self.reply({"segments": segs, "ms": round((time.time() - start) * 1000)})
                elif self.path == "/predict":
                    self.reply({"candidates": engine.predict(body["kana"], int(body.get("n", 6))),
                                "ms": round((time.time() - start) * 1000)})
                elif self.path == "/forget":
                    engine.store.forget()
                    self.reply({"ok": True})
                elif self.path == "/status":
                    self.reply(status())
                elif self.path == "/learn":
                    self.reply({"ok": engine.store.learn(body["kana"], body["surface"])})
                else:
                    self.reply({"error": "not found"}, 404)
            except Exception as e:  # 入力メソッドを止めないよう、失敗は返事で知らせる
                self.reply({"error": repr(e)}, 500)

        def do_GET(self):
            if not self.allowed(post=False):
                return
            if self.path == "/status":
                self.reply(status())
            else:
                self.reply({"error": "not found"}, 404)

        def log_message(self, *a):
            pass

    print(json.dumps(status(), ensure_ascii=False), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
