"""ツール呼び出し（read、ls、grep）の学習データを、プログラムで組み立てる。

例: make_tool_sft.py --count 3000 --out ~/ixv-lm-data/sft/tool-v1.jsonl
    make_tool_sft.py --preview 5     # 文の型がまだなくても、形だけ見る

台帳で承認済みの文章（SDD のリポジトリ、hdkworks のブログ、日本語 Wikipedia）を、仮のフォルダに並べ、
ツールを実際に走らせて結果を得る。結果の形は IXV のハーネスのツールに合わせる。
ユーザーの依頼と答えの文は corpus/tool_sft_templates.toml の型から作る（人が書く。LLM には書かせない）。
このファイルは公開していない。書き方は docs/tool-templates-guide.md。
会話は OpenAI 形式の messages で書き、学習の書式への変換は学習の側でする。
"""
import argparse
import json
import random
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
from prepare_mix import clean_parts, iter_clean  # noqa: E402

DATA = Path.home() / "ixv-lm-data"
SDD = DATA / "raw/elvez-books/SDD"
BLOG = DATA / "raw/hdkworks-blog/posts-20261108.json"
BLOG_LLM_TITLE = re.compile(r"(論文紹介|OSS紹介)")  # corpus/clean.py と同じ。LLM で書いた記事は外す
MAX_RESULT = 1500  # ツールの結果の文字数の上限。超える問題は作らない
WORD = re.compile(r"(?<![一-龯])[一-龯]{2,4}(?![一-龯])|(?<![ァ-ヴー])[ァ-ヴ][ァ-ヴー]{2,7}(?![ァ-ヴー])|(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]{3,15}(?![A-Za-z0-9_])")
HERE_WORD = "ここ"  # 文の中で {dir} が . のときに入れる言葉（ツールの引数は . のまま）

# ツールの一覧。名前と引数だけを機械的に並べる（説明の文は入れない）
TOOLS = [
    {"name": "read", "parameters": {"path": "string", "offset": "integer?", "limit": "integer?"}},
    {"name": "ls", "parameters": {"path": "string?", "limit": "integer?"}},
    {"name": "grep", "parameters": {"pattern": "string", "path": "string?", "glob": "string?",
                                    "ignoreCase": "boolean?", "literal": "boolean?", "context": "integer?",
                                    "limit": "integer?"}},
]
SYSTEM = json.dumps(TOOLS, ensure_ascii=False)


# --- 仮のフォルダ -------------------------------------------------------------

def sdd_files():
    files = {}
    for path in sorted(SDD.rglob("*")):
        rel = path.relative_to(SDD).as_posix()
        if path.is_file() and not rel.startswith(".git") and (rel.startswith(("docs/", "examples/")) or rel == "README_ja.md"):
            files[rel] = path.read_text(encoding="utf-8")
    return files


def blog_posts():
    posts = json.loads(BLOG.read_text(encoding="utf-8"))["posts"]
    return [p for p in posts if not BLOG_LLM_TITLE.search(p["title"]) and len(p["text"].strip()) >= 200]


def wiki_articles(n, seed):
    rng = random.Random(seed)
    out = []
    for text in iter_clean(clean_parts("wikipedia-ja")):
        if rng.random() < 0.01:
            title, _, body = text.partition("\n")
            if body.strip() and "/" not in title:
                out.append((title.strip(), text))
                if len(out) >= n:
                    break
    return out


class Workspaces:
    def __init__(self, seed):
        self.sdd = sdd_files()
        self.posts = blog_posts()
        self.wiki = wiki_articles(3000, seed)

    def make(self, rng):
        kind = rng.choice(["sdd", "blog", "wiki"])
        if kind == "sdd":
            return dict(self.sdd)
        files = {}
        if kind == "blog":
            for p in rng.sample(self.posts, rng.randint(5, 40)):
                files[f"blog/{p['date'][:4]}/{p['date'][5:7]}/{p['id']}.md"] = f"# {p['title']}\n\n{p['text'].strip()}\n"
        else:
            folder = rng.choice(["wiki", "docs", "notes", "資料"])
            for title, text in rng.sample(self.wiki, rng.randint(5, 30)):
                files[f"{folder}/{title}.md"] = text.strip() + "\n"
        return files


# --- ツール（IXV のハーネスと同じ形の結果を返す） ------------------------------

def dirs_of(files):
    out = {"."}
    for path in files:
        parts = path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]))
    return sorted(out)


def tool_ls(files, path="."):
    prefix = "" if path == "." else path + "/"
    entries = set()
    for f in files:
        if f.startswith(prefix):
            head, sep, _ = f[len(prefix):].partition("/")
            entries.add(head + ("/" if sep else ""))
    return sorted(entries) or None


def tool_read(files, path, offset=None, limit=None):
    lines = files[path].split("\n")
    start = (offset or 1) - 1
    end = len(lines) if limit is None else min(len(lines), start + limit)
    text = "\n".join(lines[start:end])
    if end < len(lines):
        text += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
    return text


def tool_grep(files, pattern, path="."):
    prefix = "" if path == "." else path + "/"
    out = []
    for f in sorted(files):
        if not f.startswith(prefix):
            continue
        for i, line in enumerate(files[f].split("\n"), 1):
            if pattern in line:
                out.append((f[len(prefix):], i, line if len(line) <= 200 else line[:200] + "..."))
    return out


def grep_text(matches):
    return "\n".join(f"{f}:{i}: {line}" for f, i, line in matches) if matches else "No matches found"


# --- 問題を作る ---------------------------------------------------------------

def task_ls(rng, files, kind):
    d = rng.choice(dirs_of(files))
    entries = tool_ls(files, d)
    result = "\n".join(entries)
    if len(result) > MAX_RESULT:
        return None
    slots = {"dir": d, "n": len(entries), "entries": "、".join(e.rstrip("/") for e in entries)}
    return slots, ("ls", {"path": d}), result


def task_ls_count(rng, files, kind):
    """数える問題は、中身が 12 個以下のフォルダを多めにし、答えの数が偏らないよう、数を先に均等に選ぶ
    （r2 は数え間違いが多く、1 個と言いがちだった）。"""
    global COUNT_TARGET
    by_n = {}
    for d in dirs_of(files):
        n = len(tool_ls(files, d) or [])
        if 1 <= n <= 12:
            by_n.setdefault(n, []).append(d)
    if COUNT_TARGET is None:
        COUNT_TARGET = rng.randint(1, 12) if rng.random() < 0.8 else 0
    if COUNT_TARGET:
        if COUNT_TARGET not in by_n:
            return None  # その数のフォルダがなければ、別の仮のフォルダで作り直す（狙う数は変えない）
        d = rng.choice(by_n[COUNT_TARGET])
        COUNT_TARGET = None
        entries = tool_ls(files, d)
        slots = {"dir": d, "n": len(entries), "entries": "、".join(e.rstrip("/") for e in entries)}
        return slots, ("ls", {"path": d}), "\n".join(entries)
    COUNT_TARGET = None
    return task_ls(rng, files, kind)


COUNT_TARGET = None  # 数える問題で狙っている数（作れるまで持ち越す。0 はふつうの選び方）


def task_read_first_line(rng, files, kind):
    path = rng.choice(sorted(files))
    lines = files[path].split("\n")
    at = next((i for i, l in enumerate(lines) if l.strip()), None)
    if at is None or lines[at].lstrip().startswith("<"):
        return None  # 先頭がコメント（<!-- など）のファイルは使わない
    # limit は 1〜10 に散らす（いつも 5 だと、行を指定するときも 5 をまねる）。最初の行が入る長さにはする
    limit = max(at + 1, rng.randint(1, 10))
    result = tool_read(files, path, limit=limit)
    if len(result) > MAX_RESULT:
        return None
    return {"path": path, "first_line": lines[at].strip()}, ("read", {"path": path, "limit": limit}), result


def task_read_lines(rng, files, kind):
    path = rng.choice(sorted(files))
    lines = files[path].split("\n")
    if len(lines) < 5:
        return None
    # 行数（limit）を 1〜30 に散らす。r2 は limit を 5 にしがちだった（read_first_line の limit=5 のまね）
    start = rng.randint(1, len(lines) - 1)
    end = min(len(lines), start + rng.randint(0, 29))
    result = tool_read(files, path, offset=start, limit=end - start + 1)
    content = "\n".join(lines[start - 1:end]).strip()
    if not content or len(result) > MAX_RESULT:
        return None
    return {"path": path, "start": start, "end": end, "content": content}, \
        ("read", {"path": path, "offset": start, "limit": end - start + 1}), result


def task_grep_files(rng, files, kind):
    path = rng.choice(sorted(files))
    words = WORD.findall(files[path])
    if not words:
        return None
    word = rng.choice(words)
    d = rng.choice([x for x in dirs_of(files) if x == "." or path.startswith(x + "/")])
    matches = tool_grep(files, word, d)
    result = grep_text(matches)
    if len(result) > MAX_RESULT:
        return None
    found = sorted({f for f, _, _ in matches})
    return {"word": word, "dir": d, "files": "、".join(found), "n": len(found)}, \
        ("grep", {"pattern": word, "path": d, "literal": True}), result


def task_grep_none(rng, files, kind, other):
    words = [w for w in WORD.findall(other) if not any(w in t for t in files.values())]
    if not words:
        return None
    word = rng.choice(words)
    d = rng.choice(dirs_of(files))
    return {"word": word, "dir": d}, ("grep", {"pattern": word, "path": d, "literal": True}), "No matches found"


TASKS = {
    "ls_count": task_ls_count, "ls_list": task_ls, "read_first_line": task_read_first_line,
    "read_lines": task_read_lines, "grep_files": task_grep_files, "grep_none": task_grep_none,
}


# 作業を選ぶ重み。r2 で弱かった行の指定と数えるを多めにする（tool-v2、2026-11-14）
TASK_WEIGHTS = {"ls_count": 2, "ls_list": 1, "read_first_line": 1, "read_lines": 3, "grep_files": 1, "grep_none": 1.5}

SLOTS = {  # 作業ごとに、依頼と答えで使える穴
    "ls_count": ({"dir"}, {"dir", "n"}),
    "ls_list": ({"dir"}, {"dir", "n", "entries"}),
    "read_first_line": ({"path"}, {"path", "first_line"}),
    "read_lines": ({"path", "start", "end"}, {"path", "start", "end", "content"}),
    "grep_files": ({"word", "dir"}, {"word", "dir", "files", "n"}),
    "grep_none": ({"word", "dir"}, {"word", "dir"}),
}


def check_templates(templates):
    """文の型の穴が、その作業で使えるものだけかを確かめる。問題のリストを返す。"""
    import string
    problems = []
    for task, (req_slots, ans_slots) in SLOTS.items():
        entry = templates.get(task, {})
        for kind, allowed in (("request", req_slots), ("answer", ans_slots)):
            items = entry.get(kind, [])
            if not items:
                problems.append(f"{task}.{kind}: まだ1つもありません")
            for t in items:
                try:
                    names = {f for _, f, _, _ in string.Formatter().parse(t) if f is not None}
                except ValueError as e:
                    problems.append(f"{task}.{kind}: 「{t}」の {{}} の書き方が崩れています（{e}）")
                    continue
                bad = names - allowed
                if bad:
                    problems.append(f"{task}.{kind}: 「{t}」の {{{'}、{'.join(sorted(bad))}}} はこの作業では使えません"
                                    f"（使えるのは {'、'.join('{' + s + '}' for s in sorted(allowed))}）")
    return problems


def fill(template, slots):
    if slots.get("dir") == ".":
        slots = {**slots, "dir": HERE_WORD}
        template = template.replace("{dir} ", "{dir}")  # 「ここ のファイル」の空白を詰める
    return template.format(**slots)


def pick_request(rng, templates, slots):
    """依頼の型を選ぶ。{dir} を書いていない型（「まず、確認して」など）は、いまいる場所（.）のときだけ使う。"""
    usable = [t for t in templates if "{dir}" in t or slots.get("dir", ".") == "."]
    return rng.choice(usable) if usable else None


def conversation(task, slots, call, result, templates, rng, preview):
    name, arguments = call
    if preview:
        request = f"[{task}] " + " ".join(f"{k}={v}" for k, v in slots.items() if k in ("dir", "path", "start", "end", "word"))
        answer = f"[{task} の答え] " + " ".join(f"{k}={v}" for k, v in slots.items())
    else:
        template = pick_request(rng, templates[task]["request"], slots)
        if template is None:
            return None
        request = fill(template, slots)
        answer = fill(rng.choice(templates[task]["answer"]), slots)
    # slots は採点（train/eval_tools.py）で使う。追加学習は messages だけを読む
    return {"task": task, "slots": {k: str(v) for k, v in slots.items()}, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": request},
        {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": arguments}]},
        {"role": "tool", "name": name, "content": result},
        {"role": "assistant", "content": answer},
    ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--templates", type=Path, default=HERE / "tool_sft_templates.toml")
    parser.add_argument("--preview", type=int, default=0, help="文の型を使わずに、この数だけ形を表示する")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="文の型を確かめ、数件だけ作って見せる")
    parser.add_argument("--keep-files", action="store_true",
                        help="問題ごとに仮のフォルダも残す（採点の通しで、モデル自身の呼び出しを動かすため）")
    args = parser.parse_args()

    try:
        templates = {} if args.preview else tomllib.loads(args.templates.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"{args.templates} がありません。このファイルは公開していません（自社で書いた文の型のため）。書き方は docs/tool-templates-guide.md を見て、自分で書いてください。--preview なら型なしで形だけ見られます")
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{args.templates} の書き方が崩れています: {e}")
    if args.check:
        problems = check_templates(templates)
        for p in problems:
            print("・" + p)
        if problems:
            raise SystemExit(f"直すところが {len(problems)} か所あります")
        print("文の型は大丈夫です。試しに作った会話:")
        rng = random.Random(args.seed)
        spaces = Workspaces(args.seed)
        for task in TASKS:
            for _ in range(50):
                files = spaces.make(rng)
                made = (task_grep_none(rng, files, None, rng.choice(list(spaces.make(rng).values())))
                        if task == "grep_none" else TASKS[task](rng, files, None))
                conv = conversation(task, *made, templates, rng, False) if made else None
                if conv:
                    c = conv["messages"]
                    print(f"\n[{task}]\nユーザー：{c[1]['content']}\nツール：{json.dumps(c[2]['tool_calls'][0], ensure_ascii=False)}"
                          f"\n結果：{c[3]['content'][:200]}\nアシスタント：{c[4]['content'][:300]}")
                    break
        return
    if not args.preview:
        problems = check_templates(templates)
        if problems:
            raise SystemExit("文の型に直すところがあります。--check で確かめてください")
        if not args.out:
            raise SystemExit("--out を指定してください")

    rng = random.Random(args.seed)
    spaces = Workspaces(args.seed)
    count = args.preview or args.count
    out, tries, task = [], 0, None
    while len(out) < count:
        tries += 1
        if tries > count * 50:
            raise SystemExit("問題が作れません")
        if task is None:  # 作れなかったときは同じ作業で作り直す（作業の割合を重みどおりに保つ）
            task = rng.choices(list(TASKS), weights=[TASK_WEIGHTS[t] for t in TASKS])[0]
        files = spaces.make(rng)
        if task == "grep_none":
            other = spaces.make(rng)
            made = task_grep_none(rng, files, None, rng.choice(list(other.values())))
        else:
            made = TASKS[task](rng, files, None)
        conv = conversation(task, *made, templates, rng, bool(args.preview)) if made else None
        if conv:
            if args.keep_files:  # SDD は大きいので名前だけ残す（採点のときに読み直す）
                conv["files"] = "sdd" if files == spaces.sdd else files
            out.append(conv)
            task = None

    if args.preview:
        for c in out:
            print(json.dumps(c, ensure_ascii=False, indent=1)[:2500])
            print("---")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tasks = {t: sum(c["task"] == t for c in out) for t in TASKS}
    print(json.dumps({"out": str(args.out), "count": len(out), "tasks": tasks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
