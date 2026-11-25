"""かな漢字変換の中身。辞書で下書きを作り、モデルで検算して、文節ごとの候補を出す。

進め方（azooKey の Zenzai を手本にした。コードは借りていない）：
1. 辞書（corpus/make_kanji_dict.py）の語を読みで引き、2-gram の確からしさで、読みに合う並びをビームで探す（下書き）。
2. モデル（段 1）が下書きを 1 回の計算で検算する。直した案は、モデルから見て直す前より確からしいときだけ採る。
   最初に食い違った位置で、モデルがいちばん確からしいと見る字を条件にして、
   「そこまでをこの字で始めよ」と辞書の探索をやり直す。そろうまで繰り返す（上限あり）。
   下書きは辞書の読みに縛られているので、読みに合わない字は出ない。
3. 下書きを文節に分け、文節ごとに候補を並べる。候補は：学習した変換 → 辞書の n-best → モデルが書いた変換のうち
   読みの合うもの（Sudachi で読み直して確かめる）→ ひらがな → カタカナ。
文節の区切りは、呼ぶ側が決め直せる（segments に各文節のかなの長さを渡す）。
学習（選んだ候補を覚える）とユーザー辞書は、~/Library/Application Support/IXVKanji/ の TSV に置く。
"""
import bisect
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sudachipy import Dictionary, SplitMode
from tokenizers import Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from model import LM, ModelConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dictfile  # noqa: E402

HIRA_TO_KATA = {chr(c): chr(c + 0x60) for c in range(0x3041, 0x3097)}
KATA_TO_HIRA = {v: k for k, v in HIRA_TO_KATA.items()}
ATTACH = {"助詞", "助動詞", "接尾辞", "補助記号"}  # 前の語にくっつけて 1 文節にする品詞
MAX_WORD = 16       # 辞書を引く読みの長さの上限
UNKNOWN = -16.0     # 辞書に無い 1 字の確からしさ（対数）
MLX_CACHE_MB = 32   # MLX が取っておくメモリの上限（MB）
DIGITS = re.compile(r"[0-9０-９]+")
ALNUM = re.compile(r"[A-Za-z0-9０-９Ａ-Ｚａ-ｚ]+")
KATA_RUN = re.compile(r"[ァ-ヶー]+")


def kata(s):
    return "".join(HIRA_TO_KATA.get(c, c) for c in s)


def hira(s):
    return "".join(KATA_TO_HIRA.get(c, c) for c in s)


def kanji_number(n):
    """123 → 百二十三。1 京未満。"""
    if n == 0:
        return "〇"
    digits = "〇一二三四五六七八九"
    out = ""
    for unit_value, unit in [(10 ** 12, "兆"), (10 ** 8, "億"), (10 ** 4, "万"), (1, "")]:
        part, n = divmod(n, unit_value)
        if not part:
            continue
        text = ""
        for v, name in [(1000, "千"), (100, "百"), (10, "十"), (1, "")]:
            d, part = divmod(part, v)
            if d:
                text += ("" if d == 1 and name else digits[d]) + name
        out += text + unit
    return out


# 数の読み。(読み, 値, 種類)。種類 d は 1 桁、u は 十・百・千、b は 万・億・兆
NUMBER_WORDS = [
    ("ゼロ", 0, "d"), ("レイ", 0, "d"), ("イチ", 1, "d"), ("イッ", 1, "d"), ("ニ", 2, "d"), ("サン", 3, "d"),
    ("ヨン", 4, "d"), ("ゴ", 5, "d"), ("ロク", 6, "d"), ("ロッ", 6, "d"),
    ("ナナ", 7, "d"), ("シチ", 7, "d"), ("ハチ", 8, "d"), ("ハッ", 8, "d"), ("キュウ", 9, "d"), ("ク", 9, "d"),
    ("ジュウ", 10, "u"), ("ジュッ", 10, "u"), ("ジッ", 10, "u"), ("ヒャク", 100, "u"), ("ビャク", 100, "u"),
    ("ピャク", 100, "u"), ("セン", 1000, "u"), ("ゼン", 1000, "u"),
    ("マン", 10 ** 4, "b"), ("オク", 10 ** 8, "b"), ("チョウ", 10 ** 12, "b"),
]
# 濁る・半濁る位の読みと、その前に来てよい数（さんぜん、さんびゃく、ろっぴゃく、はっぴゃく）
VOICED_UNITS = {"ゼン": {3}, "ビャク": {3}, "ピャク": {6, 8}}


def read_numbers(kana, start):
    """kana[start:] の頭を数の読みとして読む。[(終わりの位置, 値)]。"""
    found = []

    def walk(pos, total, block, digit, words, has_unit):
        if words and digit is None:
            found.append((pos, total + block))
        for reading, value, kind in NUMBER_WORDS:
            if not kana.startswith(reading, pos):
                continue
            end = pos + len(reading)
            if kind == "d" and digit is None:
                if value == 0 and (words or total or block):
                    continue
                if end == len(kana) or not any(kana.startswith(r, end) and k != "d" for r, _, k in NUMBER_WORDS):
                    found.append((end, total + block + value))  # 一の位として終わる
                walk(end, total, block, value, words + 1, has_unit)
            elif kind == "u":
                if reading in VOICED_UNITS and digit not in VOICED_UNITS[reading]:
                    continue  # ゼン・ビャク・ピャクは、サン・ロッ・ハッの後だけ（「ぜんかく」を 1000 角にしない）
                d = 1 if digit is None else digit
                if block % (value * 10) and block % (value * 10) >= value:
                    continue
                walk(end, total, block + d * value, None, words + 1, True)
            elif kind == "b":
                part = block + (digit or 0)
                if not part or total % (value * 10 ** 4 if value < 10 ** 12 else 10 ** 16) >= value:
                    continue
                walk(end, total + part * value, 0, None, words + 1, True)

    walk(start, 0, 0, None, 0, False)
    return sorted(set(found))


def number_forms(s):
    half = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    full = half.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    forms = [full, half, "".join("〇一二三四五六七八九"[int(c)] for c in half)]
    if len(half) <= 16:
        forms.append(kanji_number(int(half)))
        if len(half) > 3:
            forms.append(f"{int(half):,}")
    return list(dict.fromkeys(forms))


SYMBOLS = {
    "キゴウ": list("※→←↑↓○●◎△▲▽▼□■◇◆☆★♪〒〜…・：；＊＃＆＠§¶†‡"),
    "ヤジルシ": list("→←↑↓⇒⇔↔"),
    "マル": list("○●◎①②③④⑤⑥⑦⑧⑨⑩"),
    "シカク": list("□■◇◆"),
    "ホシ": list("☆★"),
    "カッコ": ["「」", "『』", "（）", "【】", "［］", "〈〉", "《》", "〔〕"],
    "ユウビン": ["〒"],
    "オンプ": ["♪", "♫"],
    "ドウジョウ": ["〃", "々"],
    "コメ": ["※"],
}


def wareki(d):
    return f"令和{d.year - 2018}年{d.month}月{d.day}日" if d.year >= 2019 else None


def special_candidates(kana, now=None):
    """日付・時刻・記号の候補。きょう、あした、きのう、いま、きごう など。"""
    now = now or datetime.datetime.now()
    days = {"キョウ": 0, "ホンジツ": 0, "アシタ": 1, "アス": 1, "ミョウニチ": 1, "キノウ": -1, "サクジツ": -1,
            "アサッテ": 2, "オトトイ": -2}
    out = []
    if kana in days:
        d = (now + datetime.timedelta(days=days[kana])).date()
        week = "月火水木金土日"[d.weekday()]
        out += [f"{d.year}年{d.month}月{d.day}日", f"{d.month}月{d.day}日（{week}）", f"{d.month}月{d.day}日",
                f"{d.year}/{d.month:02d}/{d.day:02d}", f"{d.year}-{d.month:02d}-{d.day:02d}", wareki(d)]
    if kana in ("イマ", "ジコク", "ゲンザイ"):
        out += [f"{now.hour}時{now.minute:02d}分", f"{now.hour:02d}:{now.minute:02d}",
                f"{now.month}月{now.day}日 {now.hour}時{now.minute:02d}分"]
    if kana in ("コトシ", "ホンネン"):
        out += [f"{now.year}年", f"令和{now.year - 2018}年"]
    out += SYMBOLS.get(kana, [])
    return [c for c in out if c]


USER_HEADER = "# 読み（ひらがなかカタカナ）<TAB>変換。# で始まる行は飛ばす。UTF-8 で保存する\n"
MAX_LEARN = 64  # 覚える読みと変換の長さの上限（字）
SHORT_READING = 2  # この字数までの読み（ね、が など）は、覚えたものを弱く効かせる（elvezjp/ixv-lm#11）


def learned_bonus(kana, n):
    """覚えた語の点。何回覚えても、ユーザー辞書の語（-1.0）は超えない。短い読みは、ほかの文に響かないよう弱くする。"""
    return (-4.0 if len(kana) <= SHORT_READING else -2.0) + min(math.log1p(n), 0.9)


def read_lines(path, name):
    """TSV を行ごとに読む。UTF-8 で読めない行は飛ばし、数をログに出す（elvezjp/ixv-lm#6）。"""
    lines, bad = [], 0
    for raw in path.read_bytes().splitlines():
        try:
            lines.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            bad += 1
    if bad:
        print(f"{name}: UTF-8 で読めない行を {bad} 行飛ばしました（{path}）", file=sys.stderr, flush=True)
    return lines


class Store:
    """学習とユーザー辞書。TSV のファイルを、変わっていたら読み直す。
    壊れた行やファイルがあっても、変換は止めない（elvezjp/ixv-lm#5、#6）。"""

    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.history_path = folder / "history.tsv"
        self.user_path = folder / "user_dict.tsv"
        if not self.user_path.exists():
            self.user_path.write_text(USER_HEADER, encoding="utf-8")
        self.history, self.history_mtime = defaultdict(lambda: defaultdict(int)), None
        self.user, self.user_mtime = defaultdict(list), None
        self.enabled = True  # 学習を使うか（設定画面から、変換のたびに渡される）
        self.reload_history()

    @staticmethod
    def mtime(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def reload_history(self):
        """history.tsv が変わっていたら読み直す。手で直した記録も、動かし直さずに効く（elvezjp/ixv-lm#11）。"""
        mtime = self.mtime(self.history_path)
        if mtime == self.history_mtime:
            return
        history, bad = defaultdict(lambda: defaultdict(int)), 0
        if mtime is not None:
            for line in read_lines(self.history_path, "history.tsv"):
                parts = line.split("\t")
                try:
                    kana, surface, n = parts
                    history[kana][surface] += int(n)
                except ValueError:
                    bad += 1
        if bad:
            print(f"history.tsv: 読めない行を {bad} 行飛ばしました", file=sys.stderr, flush=True)
        self.history, self.history_mtime = history, mtime

    def user_dict(self):
        mtime = self.mtime(self.user_path)
        if mtime is None:  # 消されたら、空の辞書として作り直す
            try:
                self.user_path.write_text(USER_HEADER, encoding="utf-8")
                mtime = self.mtime(self.user_path)
            except OSError:
                self.user, self.user_mtime = defaultdict(list), None
                return self.user
        if mtime != self.user_mtime:
            user = defaultdict(list)
            try:
                lines = read_lines(self.user_path, "user_dict.tsv")
            except OSError:
                lines = []
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 2 and not line.startswith("#") and parts[0].strip() and parts[1].strip():
                    user[kata(parts[0].strip())].append(parts[1].strip())
            self.user, self.user_mtime = user, mtime
        return self.user

    def learn(self, kana, surface):
        """覚える。タブや改行の入ったもの、長すぎるもの、空のものは捨てる（elvezjp/ixv-lm#5）。"""
        kana, surface = kata(str(kana)), str(surface)
        if not kana or not surface or len(kana) > MAX_LEARN or len(surface) > MAX_LEARN \
                or any(c in s for s in (kana, surface) for c in "\t\r\n"):
            return False
        self.reload_history()
        self.history[kana][surface] += 1
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(f"{kana}\t{surface}\t1\n")
        self.history_mtime = self.mtime(self.history_path)
        return True

    def forget(self):
        """学習を全部消す。"""
        self.history.clear()
        if self.history_path.exists():
            self.history_path.unlink()
        self.history_mtime = None

    def learned(self, kana):
        if not self.enabled:
            return []
        return sorted(self.history.get(kata(kana), {}).items(), key=lambda x: -x[1])


class Engine:
    def __init__(self, dict_dir, run, data, store_dir, ckpt=None, verify_rounds=6, verify_margin=1.0,
                 cache=dictfile.CACHE, bits=None, mlx_cache_mb=MLX_CACHE_MB):
        self.verify_rounds = verify_rounds
        self.cache = cache
        self.bits = bits
        self.verify_margin = verify_margin  # 直した案は、モデルの確からしさがこれだけ（対数で）上回るときだけ採る
        self.load_dict(dict_dir)
        self.tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
        meta = json.loads((data / "meta.json").read_text())
        assert meta.get("format") == "kanji-pairs", "段 1 の文字単位のデータを渡す"
        self.eot = meta["eot"]
        config = json.loads((run / "config.json").read_text())
        self.model = LM(ModelConfig(**config["model"]))
        if (run / "model.safetensors").exists():  # 配る形（export_release.py）。重みがフォルダの直下にある
            self.ckpt = json.loads((run / "config.json").read_text()).get("ckpt", "")
            self.model.load_weights(str(run / "model.safetensors"))
        else:
            self.ckpt = ckpt or (run / "latest").read_text().strip()
            self.model.load_weights(str(run / self.ckpt / "model.safetensors"))
        if bits:  # 線形層の重みを bits ビットにする。埋め込み（出力にも使う）はそのまま（elvezjp/ixv-lm#1）
            nn.quantize(self.model, group_size=64, bits=bits, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        mx.eval(self.model.parameters())
        # MLX は使い終えたメモリを、次の計算のために取っておく。取っておく量に上限を付ける（elvezjp/ixv-lm#1）
        mx.set_cache_limit(mlx_cache_mb * 2 ** 20)
        mx.clear_cache()
        self.sudachi = Dictionary(dict="core").tokenizer(mode=SplitMode.C)
        self.store = Store(store_dir)

    # ---------- 辞書 ----------
    def load_dict(self, folder):
        """辞書を、mmap で開けるファイルの形（ime/dictfile.py）にして開く。無ければ初回だけ作る（elvezjp/ixv-lm#1）。"""
        self.dict = dictfile.Dictionary(dictfile.ensure(folder, self.cache))
        for name in ("n_words", "surface", "reading", "pos", "count", "logp_uni", "canon", "by_reading",
                     "predict_word", "predict_count", "predict_keys", "follow_types", "follow_total"):
            setattr(self, name, getattr(self.dict, name))

    def logp(self, prev, word):
        """Witten-Bell で 1-gram と混ぜた 2-gram の確からしさ（対数）。"""
        prev = self.canon[prev]
        t, c = self.follow_types[prev], self.follow_total[prev]
        uni = math.exp(self.logp_uni[word])
        if c == 0:
            return math.log(uni)
        count = self.dict.bigram(prev, word)
        if count is None:
            share = self.count[word] / max(self.count[self.canon[word]], 1)
            count = self.dict.bigram(prev, self.canon[word], 0) * min(share, 1.0)
        return math.log((count + t * uni) / (c + t))

    def lookup(self, kana, start):
        """kana[start:] の頭に合う語。(終わりの位置, 語 id か None, 表層, 対数の確からしさの足し) の並び。"""
        found = []
        user = self.store.user_dict()
        digit = DIGITS.match(kana, start)
        if digit:  # 数字はひとかたまり
            for k, form in enumerate(number_forms(digit.group())):
                found.append((digit.end(), None, form, -2.0 - k))
            return found
        alnum = ALNUM.match(kana, start)
        if alnum:
            found.append((alnum.end(), None, alnum.group(), -3.0))
            return found
        for end, value in read_numbers(kana, start):  # 数の読み（サンビャク → ３００）
            base = -12.0 if value < 10 else -7.0  # 1 字の数（サン、ニ）は、ふつうの語に負けやすく
            for k, form in enumerate(number_forms(str(value))):
                found.append((end, None, form, base - k))
        run = KATA_RUN.match(kana, start)  # 辞書に無いカタカナ語は、カタカナのまま
        if run:
            for end in range(start + 2, min(run.end(), start + MAX_WORD) + 1):
                found.append((end, None, kana[start:end], -8.0 - 6.0 * (end - start)))
        for end in range(start + 1, min(len(kana), start + MAX_WORD) + 1):
            piece = kana[start:end]
            for i in self.by_reading.get(piece, ()):
                found.append((end, i, self.surface[i], 0.0))
            for s in user.get(piece, ()):  # ユーザー辞書は、使う人がはっきり書いたものなので、覚えたものより上
                found.append((end, None, s, -1.0))
            for s, n in self.store.learned(piece)[:3]:
                found.append((end, None, s, learned_bonus(piece, n)))
        found.append((start + 1, None, hira(kana[start]), UNKNOWN))
        return found

    def search(self, kana, prev=0, prefix="", beam=40, n=10):
        """読み kana に合う並びをビームで探す。prefix があれば、表層がそれで始まるものだけ。
        返すのは [(確からしさ, [(表層, 語 id, 読みの長さ)])] の上位 n。"""
        states = defaultdict(list)  # 位置 → [(確からしさ, 表層, 最後の語 id, 道筋)]
        states[0] = [(0.0, "", prev, ())]
        logp = {}  # 同じ (前の語, 語) は 1 回だけ計算する
        for pos in range(len(kana)):
            if not states[pos]:
                continue
            found = self.lookup(kana, pos)  # 同じ位置の辞書引きは、ビームのどの道筋でも同じ
            for score, text, last, path in sorted(states[pos], key=lambda x: -x[0])[:beam]:
                for end, word, surface, bonus in found:
                    new_text = text + surface
                    if prefix and not (new_text.startswith(prefix) or prefix.startswith(new_text)):
                        continue
                    if word is None:
                        step = bonus
                        nxt = 0
                    else:
                        pair = (last, word)
                        if pair not in logp:
                            logp[pair] = self.logp(last, word)
                        step = logp[pair] + bonus
                        nxt = word
                    states[end].append((score + step, new_text, nxt, path + ((surface, word, end - pos),)))
            del states[pos]
        final = {}
        for score, text, last, path in states[len(kana)]:
            if prefix and not text.startswith(prefix):
                continue
            score += self.logp(last, self.by_reading["。"][0]) if last and self.by_reading.get("。") else 0.0
            if text not in final or final[text][0] < score:
                final[text] = (score, path)
        return sorted(final.values(), key=lambda x: -x[0])[:n]

    # ---------- モデル ----------
    def prompt(self, context, kana):
        return self.tokenizer.encode(f"<|ctx|>{context[-40:]}<|read|>{kana}<|out|>").ids

    def model_check(self, context, kana, draft, start=0):
        """下書きを 1 回の計算で検算する。start 字目より後ろで最初に食い違った位置と、モデルの字を返す（そろえば None）。"""
        prompt = self.prompt(context, kana)
        ids = self.tokenizer.encode(draft).ids
        unk = self.tokenizer.token_to_id("<|unk|>")
        logits = self.model(mx.array([prompt + ids]))[0, len(prompt) - 1:]
        best = mx.argmax(logits, axis=-1).tolist()
        pos = 0
        for i, token in enumerate(ids):
            piece = self.tokenizer.decode([token])
            if pos >= start and token != unk and best[i] != token:
                guess = self.tokenizer.decode([best[i]])
                return pos, guess if best[i] != self.eot else None
            pos += len(piece)
        return None

    def model_score(self, context, kana, text):
        """モデルから見た text の確からしさ（対数、終わりの印まで）。"""
        prompt = self.prompt(context, kana)
        ids = self.tokenizer.encode(text).ids + [self.eot]
        logits = self.model(mx.array([prompt + ids[:-1]]))[0, len(prompt) - 1:].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.take_along_axis(logp, mx.array(ids)[:, None], axis=-1).sum().item()

    def model_candidates(self, context, kana, n=5):
        """モデルに書かせた変換をビームで n 個。読みは縛っていない。"""
        prompt = self.prompt(context, kana)
        beams, done = [([], 0.0)], []
        for _ in range(len(kana) + 4):
            batch = mx.array([prompt + out for out, _ in beams])
            logp = mx.log(mx.softmax(self.model(batch)[:, -1].astype(mx.float32), axis=-1))
            top = mx.argsort(-logp, axis=-1)[:, :n].tolist()
            vals = logp.tolist()
            grown = sorted(((out + [t], s + vals[b][t]) for b, (out, s) in enumerate(beams) for t in top[b]),
                           key=lambda x: -x[1])
            beams = []
            for out, s in grown:
                if out[-1] == self.eot:
                    done.append((out[:-1], s))
                elif len(beams) < n:
                    beams.append((out, s))
            if not beams or len(done) >= n:
                break
        texts = []
        for out, _ in sorted(done + beams, key=lambda x: -x[1]):
            t = self.tokenizer.decode(out)
            if t and t not in texts:
                texts.append(t)
        return texts

    def reads_as(self, text, kana):
        """text を Sudachi で読み直すと kana になるか。"""
        if ALNUM.search(text):
            return False
        reading = "".join(m.surface() if m.part_of_speech()[0] in ("補助記号", "空白") else m.reading_form()
                          for m in self.sudachi.tokenize(text))
        return kata(reading) == kana

    # ---------- 全体 ----------
    def draft(self, context, kana):
        """辞書の下書きを、モデルの検算でそろえる。返すのは道筋（(表層, 語 id, 読みの長さ) の並び）。"""
        found = self.search(kana, n=1)
        if not found:
            return [(hira(kana), None, len(kana))]
        path = found[0][1]
        start = 0  # これより前の食い違いは、直した案を採らなかったもの
        for _ in range(self.verify_rounds):
            text = "".join(p[0] for p in path)
            check = self.model_check(context, kana, text, start)
            if check is None or check[1] is None:
                break
            pos, char = check
            again = self.search(kana, prefix=text[:pos] + char, n=1)
            new_text = "".join(p[0] for p in again[0][1]) if again else None
            if new_text is None or \
                    self.model_score(context, kana, new_text) <= self.model_score(context, kana, text) + self.verify_margin:
                # 直した案のほうが、モデルから見ても確からしくなければ採らない。
                # 検算はやめずに、後ろの食い違いへ進む（表記の揺れで止まって、文末の間違いが残らないように。elvezjp/ixv-lm#3）
                start = pos + 1
                continue
            path = again[0][1]
        return path

    def bunsetsu(self, path):
        """語の並びを文節に分ける。返すのは [(かなの長さ, 下書きの表層)]。"""
        segs = []
        for surface, word, length in path:
            attach = word is not None and self.pos[word] in ATTACH
            if segs and attach:
                segs[-1] = (segs[-1][0] + length, segs[-1][1] + surface)
            else:
                segs.append((length, surface))
        return segs

    def predict(self, kana, n=6):
        """読みが kana で始まる語（学習したものを先に、あとは回数の多い順）。打っている途中の予測変換。"""
        kana = kata(kana)
        found = defaultdict(float)
        for k, items in (self.store.history.items() if self.store.enabled else ()):
            if k.startswith(kana) and len(k) > len(kana):
                for surface, c in items.items():
                    found[surface] += 1e9 * c
        lo = bisect.bisect_left(self.predict_keys, kana)
        hi = bisect.bisect_left(self.predict_keys, kana + "\uffff")
        for j in range(lo, hi):
            word = self.predict_word[j]
            if len(self.reading[word]) > len(kana):
                found[self.surface[word]] += self.predict_count[j]
        return [s for s, _ in sorted(found.items(), key=lambda x: -x[1])[:n]]

    def segment_candidates(self, context, kana, prev_text, n=9, fast=False):
        cands = list(self.store.user_dict().get(kana, []))
        cands += [s for s, _ in self.store.learned(kana)]
        for score, path in self.search(kana, n=n):
            cands.append("".join(p[0] for p in path))
        if not fast:
            for t in self.model_candidates(context + prev_text, kana, n=5):
                if self.reads_as(t, kana):
                    cands.append(t)
        cands += [hira(kana), kana]
        brackets = set("（）()「」『』【】[]")
        cands = [c for c in cands if c and not (set(c) & brackets - set(kana))]
        cands[1:1] = special_candidates(kana)  # 日付・時刻・記号は、1 番目の次に
        return list(dict.fromkeys(cands))

    def convert(self, context, kana, segments=None, fast=False):
        """文節ごとの候補。segments（各文節のかなの長さ）を渡せば、その区切りで変換する。
        fast は、モデルを使わない（ライブ変換で打つたびに呼ぶ用）。"""
        kana = kata(kana)
        if segments and sum(segments) == len(kana) and all(x > 0 for x in segments):
            segs = [(length, None) for length in segments]
        else:
            if fast:
                found = self.search(kana, n=1)
                path = found[0][1] if found else [(hira(kana), None, len(kana))]
            else:
                path = self.draft(context, kana)
            segs = self.bunsetsu(path)
        result, pos, prev_text = [], 0, ""
        for length, drafted in segs:
            piece = kana[pos:pos + length]
            learned = [s for s, _ in self.store.learned(piece)]
            user = self.store.user_dict().get(piece, [])
            # 1 番目は、ユーザー辞書、覚えたもの、下書きの順。短い読みは、覚えたものを下書きより前に出さない
            first = user[:1] + (learned[:1] if len(piece) > SHORT_READING else [])
            cands = first + ([drafted] if drafted else []) + self.segment_candidates(
                context, piece, prev_text, fast=fast)
            cands = list(dict.fromkeys(cands))
            result.append({"kana": hira(piece), "candidates": cands})
            prev_text += cands[0]
            pos += length
        return result
