"""変換の辞書（words.tsv、bigrams.npy）を、mmap で開けるファイルの形にして持つ（elvezjp/ixv-lm#1）。

辞書を Python の str、list、dict で持つと、55 万語と 553 万組の 2-gram で数百 MB になる。
そこで、最初に一度だけ別のプロセスで、引くのに要る配列をそのままのバイト列でファイルに書き、
変換サーバーはそれを mmap で開く。mmap の頁は、OS がいつでも捨てて読み直せるので、
変換サーバーのメモリ（物理フットプリント）にほとんど数えられない。中身と引いた結果は、元の辞書と同じ。

置き場所は ~/Library/Caches/IXVKanji/dict-<辞書の大きさと日付から作った印>/。消しても、次の起動で作り直す。

ファイル（数は配列の要素の数）：
  surface.bin, surface.end   語の表層（UTF-8 をつなげたもの、各語の終わりの位置 int64）。n 語
  reading.bin, reading.end   語の読み。同じ形
  pos.u16                    品詞の番号。番号 → 品詞は meta.json の pos
  count.f64, logp_uni.f64    回数と、1-gram の確からしさ（対数）
  canon.i32                  同じ表層・品詞で、いちばん回数の多い読みの語 id
  order.i32                  語 id（1 から）を読みの順に並べたもの（同じ読みの中は id の順）
  ureading.bin, ureading.end 読みの種類を、読みの順に（UTF-8 のバイト順は、文字の順と同じ）
  ustart.i32                 読みの種類ごとの、order の中の始まり（最後に order の長さ）
  pred_word.i32, pred_count.f64  予測変換：（読み、表層）ごとに回数を足したもの。（読み、表層）の順
  brow.i64, bcol.i32, bcount.i32 2-gram：前の語ごとの範囲、後ろの語 id（範囲の中は小さい順）、回数
  follow_types.f64, follow_total.f64  前の語ごとの、後ろに来る語の種類の数と回数の合計
"""
import bisect
import functools
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import numpy as np

FORMAT = 1
ATTACH = {"助詞", "助動詞", "接尾辞", "補助記号"}  # engine.py と同じ。前の語にくっつけて 1 文節にする品詞
CACHE = Path.home() / "Library/Caches/IXVKanji"


def cache_folder(dict_dir, cache=CACHE):
    """辞書のファイルの大きさと日付から、置き場所を決める。辞書を入れ替えれば、別の置き場所になる。"""
    sign = [FORMAT, str(Path(dict_dir).resolve())]
    for name in ("words.tsv", "bigrams.npy"):
        st = (Path(dict_dir) / name).stat()
        sign += [name, st.st_size, st.st_mtime_ns]
    return Path(cache) / ("dict-" + hashlib.sha1(json.dumps(sign).encode()).hexdigest()[:16])


def ensure(dict_dir, cache=CACHE):
    """ファイルの形の辞書が無ければ、別のプロセスで作る（作る途中のメモリを、変換サーバーに残さない）。置き場所を返す。"""
    folder = cache_folder(dict_dir, cache)
    if not (folder / "meta.json").exists():
        print(f"辞書をファイルの形にしています（初回だけ）：{folder}", file=sys.stderr, flush=True)
        subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), str(dict_dir), str(folder)], check=True)
    return folder


def build(dict_dir, folder):
    """words.tsv と bigrams.npy から、ファイルの形の辞書を作る。途中まで作ったものは残さない。"""
    dict_dir, folder = Path(dict_dir), Path(folder)
    tmp = folder.with_name(folder.name + f".tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    def write(name, data):
        with open(tmp / name, "wb") as f:
            f.write(data if isinstance(data, (bytes, bytearray)) else memoryview(data).cast("B"))

    surfaces, readings, counts, pos_ids, pos_names = [], [], [], array("H"), {}
    with (dict_dir / "words.tsv").open(encoding="utf-8") as f:
        for line in f:
            i, s, r, p, c = line.rstrip("\n").split("\t")
            surfaces.append(s)
            readings.append(r)
            pos_ids.append(pos_names.setdefault(p, len(pos_names)))
            counts.append(int(c))
    n = len(surfaces)

    def strings(name, items):
        blob, ends = bytearray(), array("q")
        for s in items:
            blob += s.encode("utf-8")
            ends.append(len(blob))
        write(name + ".bin", blob)
        write(name + ".end", ends)

    strings("surface", surfaces)
    strings("reading", readings)
    write("pos.u16", pos_ids)
    counts = np.array(counts, dtype=np.float64)
    write("count.f64", counts.tobytes())
    write("logp_uni.f64", np.log(np.maximum(counts, 1) / counts[1:].sum()).tobytes())

    # 同じ表層・品詞で読みだけ違う語（明日：アス、アシタ）は、いちばん回数の多い読みの語に寄せる（同じ回数なら id の小さい方）
    top = {}
    for i in range(1, n):
        key = (surfaces[i], pos_ids[i])
        if key not in top or counts[i] > counts[top[key]]:
            top[key] = i
    write("canon.i32", array("i", [0] + [top[(surfaces[i], pos_ids[i])] for i in range(1, n)]))
    del top

    # 読み → 語 id
    order = sorted(range(1, n), key=readings.__getitem__)  # 安定。同じ読みの中は id の順
    write("order.i32", array("i", order))
    unique, starts = [], array("i")
    for k, i in enumerate(order):
        if not unique or readings[i] != unique[-1]:
            unique.append(readings[i])
            starts.append(k)
    starts.append(len(order))
    strings("ureading", unique)
    write("ustart.i32", starts)

    # 予測変換：読みが 2 字以上で、くっつく品詞でない語を、（読み、表層）ごとに足す
    attach = {pos_names[p] for p in ATTACH if p in pos_names}
    best = {}
    for i in range(1, n):
        if pos_ids[i] in attach or len(readings[i]) < 2:
            continue
        key = (readings[i], surfaces[i])
        if key in best:
            best[key][1] += counts[i]
        else:
            best[key] = [i, counts[i]]
    items = sorted(best.items())
    write("pred_word.i32", array("i", [w for _, (w, _) in items]))
    write("pred_count.f64", array("d", [float(c) for _, (_, c) in items]))
    del best, items

    # 2-gram：前の語の順、同じ前の語の中は後ろの語の順
    bigrams = np.load(dict_dir / "bigrams.npy", mmap_mode="r")
    prev, word, cnt = (np.asarray(bigrams[:, k]) for k in range(3))
    sort = np.lexsort((word, prev))
    if len(sort) > 1:
        same = (prev[sort][1:] == prev[sort][:-1]) & (word[sort][1:] == word[sort][:-1])
        assert not same.any(), "bigrams.npy に同じ組が 2 回ある"
    assert cnt.max(initial=0) < 2 ** 31 and n < 2 ** 31
    write("brow.i64", np.searchsorted(prev[sort], np.arange(n + 1)).astype(np.int64).tobytes())
    write("bcol.i32", word[sort].astype(np.int32).tobytes())
    write("bcount.i32", cnt[sort].astype(np.int32).tobytes())
    write("follow_types.f64", np.bincount(prev, minlength=n).astype(np.float64).tobytes())
    write("follow_total.f64", np.bincount(prev, weights=cnt, minlength=n).tobytes())

    (tmp / "meta.json").write_text(json.dumps(
        {"format": FORMAT, "n": n, "pos": list(pos_names), "source": str(dict_dir.resolve())}, ensure_ascii=False))
    try:
        tmp.rename(folder)
    except OSError:  # ほかのプロセスが先に作った
        shutil.rmtree(tmp, ignore_errors=True)


def open_map(path, fmt):
    """ファイルを mmap で開き、fmt（array と同じ記号）の並びとして返す。"""
    size = os.path.getsize(path)
    if size == 0:
        return array(fmt)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return memoryview(mm).cast(fmt) if fmt != "B" else mm


class Strings:
    """文字列の並び（mmap したバイト列と、終わりの位置）。"""

    def __init__(self, folder, name):
        self.blob = open_map(folder / (name + ".bin"), "B")
        self.ends = open_map(folder / (name + ".end"), "q")

    def __len__(self):
        return len(self.ends)

    def raw(self, i):
        return self.blob[self.ends[i - 1] if i else 0:self.ends[i]]

    def __getitem__(self, i):
        return self.raw(i).decode("utf-8")


class Raw:
    """Strings の j 番目を、UTF-8 のバイト列のまま返す並び（bisect で引くため）。"""

    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, j):
        return self.strings.raw(j)


class Coded:
    """種類の少ない文字列（品詞）の並び。"""

    def __init__(self, ids, names):
        self.ids, self.names = ids, names

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.names[self.ids[i]]


class View:
    """seq[index[j]] を j 番目とする並び（bisect で引くため）。"""

    def __init__(self, seq, index):
        self.seq, self.index = seq, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        return self.seq[self.index[j]]


class ReadingIndex:
    """読み → 語 id の並び。"""

    def __init__(self, folder):
        self.order = open_map(folder / "order.i32", "i")
        self.unique = Raw(Strings(folder, "ureading"))
        self.starts = open_map(folder / "ustart.i32", "i")
        # 同じ読みは何度も引く（打つたびのライブ変換、文節ごとの探索）。最近引いた読みの範囲だけ覚えておく
        self.find = functools.lru_cache(maxsize=1 << 15)(self._find)

    def _find(self, reading):
        key = reading.encode("utf-8")
        j = bisect.bisect_left(self.unique, key)
        if j < len(self.unique) and self.unique[j] == key:
            return self.starts[j], self.starts[j + 1]
        return None

    def get(self, reading, default=None):
        span = self.find(reading)
        return default if span is None else self.order[span[0]:span[1]]

    def __getitem__(self, reading):
        found = self.get(reading)
        if found is None:
            raise KeyError(reading)
        return found


class Dictionary:
    """ファイルの形の辞書を mmap で開いたもの。"""

    def __init__(self, folder):
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        assert meta["format"] == FORMAT
        self.n_words = meta["n"]
        self.surface = Strings(folder, "surface")
        self.reading = Strings(folder, "reading")
        self.pos = Coded(open_map(folder / "pos.u16", "H"), meta["pos"])
        self.count = open_map(folder / "count.f64", "d")
        self.logp_uni = open_map(folder / "logp_uni.f64", "d")
        self.canon = open_map(folder / "canon.i32", "i")
        self.by_reading = ReadingIndex(folder)
        self.predict_word = open_map(folder / "pred_word.i32", "i")
        self.predict_count = open_map(folder / "pred_count.f64", "d")
        self.predict_keys = View(self.reading, self.predict_word)
        self.brow = open_map(folder / "brow.i64", "q")
        self.bcol = open_map(folder / "bcol.i32", "i")
        self.bcount = open_map(folder / "bcount.i32", "i")
        self.follow_types = open_map(folder / "follow_types.f64", "d")
        self.follow_total = open_map(folder / "follow_total.f64", "d")

    def bigram(self, prev, word, default=None):
        """前の語 prev のあとに word が来た回数。無ければ default。"""
        lo, hi = self.brow[prev], self.brow[prev + 1]
        j = bisect.bisect_left(self.bcol, word, lo, hi)
        if j < hi and self.bcol[j] == word:
            return self.bcount[j]
        return default


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
