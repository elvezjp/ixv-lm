"""取得したデータを、学習用の共通形式（Parquet の text 列）に加工する。

例: clean.py aozora
出力は ~/ixv-lm-data/clean/<台帳の id>/ に part-*.parquet と source.json を置く。
どのデータにも、改行と空白の正規化、短すぎる文書の除外、同じ本文の重複除去をかける。
"""
import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import io
import json
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path.home() / "ixv-lm-data/raw"
OUT = Path.home() / "ixv-lm-data/clean"
MIN_CHARS = 20
PART_DOCS = 100_000


def normalize(text):
    # 全角と半角の区別は日本語の表記として残すので NFKC ではなく NFC にする
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# 青空文庫 ---------------------------------------------------------------

AOZORA_NOTE = re.compile(r"［＃[^］]*］")
AOZORA_GAIJI = re.compile(r"※［＃[^］]*］")
AOZORA_UNICODE = re.compile(r"U\+([0-9A-Fa-f]{4,6})")


def aozora_gaiji(match):
    # 新しい注記は Unicode の符号位置を持つ。持たない外字は落とす
    code = AOZORA_UNICODE.search(match.group(0))
    return chr(int(code.group(1), 16)) if code else ""


def aozora_text(raw):
    text = raw.decode("cp932", errors="replace")
    lines = text.replace("\r\n", "\n").split("\n")
    # 冒頭の「テキスト中に現れる記号について」の囲みを除く
    rules = [i for i, line in enumerate(lines[:60]) if re.fullmatch(r"-{20,}", line.strip())]
    if len(rules) >= 2:
        lines = lines[: rules[0]] + lines[rules[1] + 1 :]
    text = "\n".join(lines)
    text = AOZORA_GAIJI.sub(aozora_gaiji, text)
    text = re.sub(r"《[^》]*》", "", text).replace("｜", "")
    # 底本と入力者の情報は残す（青空文庫の希望）。字下げなどの注記だけを除く
    return AOZORA_NOTE.sub("", text)


def docs_aozora(stats):
    for path in sorted((RAW / "aozora/text").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".txt")]
            if len(names) != 1:
                stats["skipped_archives"] = stats.get("skipped_archives", 0) + 1
                continue
            yield path.stem, aozora_text(archive.read(names[0]))


# e-Gov 法令 -------------------------------------------------------------

LAW_LINES = {"LawNum", "LawTitle", "EnactStatement", "PartTitle", "ChapterTitle", "SectionTitle",
             "SubsectionTitle", "DivisionTitle", "ArticleCaption", "SupplProvisionLabel", "AppdxTableTitle"}
LAW_INLINE = {"ArticleTitle", "ParagraphNum", "ItemTitle", "Subitem1Title", "Subitem2Title", "Subitem3Title"}


def law_text(raw):
    root = ET.fromstring(raw)
    for reading in root.iter("Rt"):
        reading.text = ""  # ルビの読みを本文に混ぜない
    lines, current = [], []

    def flush():
        if current:
            lines.append("　".join(current))
            current.clear()

    for element in root.iter():
        tag = element.tag
        content = "".join(element.itertext()).strip()
        if tag in LAW_LINES:
            flush()
            if content:
                lines.append(content)
        elif tag in LAW_INLINE and content:
            # 第1項の項番号は空なので、条の見出しと本文を同じ行に続ける
            flush()
            current.append(content)
        elif tag == "Sentence":
            if content:
                current.append(content)
    flush()
    return "\n".join(lines)


def docs_egov(stats):
    with zipfile.ZipFile(RAW / "egov-law/all_xml.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".xml")):
            yield name.rsplit("/", 1)[-1][:-4], law_text(archive.read(name))


# Wikipedia --------------------------------------------------------------

def docs_wikipedia(stats):
    for path in sorted((RAW / "wikipedia-20231101-ja").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["id", "title", "text"]):
            for doc, title, text in zip(*(batch.column(c).to_pylist() for c in ("id", "title", "text"))):
                yield doc, f"{title}\n\n{text}"


# Wikipedia の公式ダンプ（XML、wikitext）。2026-11-24 決定（20231101 は古いので、最新を取り直す）
# HF の wikimedia/wikipedia と同じく、本文だけを残す。テンプレートと表は落とし、いくつかのテンプレートは中の語を残す。

JAWIKI_DUMP = RAW / "jawiki-20261101/jawiki-20261101-pages-articles.xml.bz2"
WIKI_KEEP_FIRST = {"仮リンク", "読み仮名", "読み仮名_ruby不使用", "ruby", "要出典範囲", "nowrap", "補助漢字フォント",
                   "jis2004フォント", "拡張漢字", "lang-en-short", "smaller", "small", "nobr", "要検証範囲"}
WIKI_KEEP_LAST = {"lang", "langwithname"}
WIKI_DROP_LINK = re.compile(r"^(ファイル|画像|file|image|category|カテゴリ|media|メディア)\s*:", re.I)
WIKI_TAG_DROP = re.compile(r"<(ref|math|gallery|timeline|score|syntaxhighlight|source|imagemap|templatedata|graph|chem|ce)"
                           r"\b[^>]*?(/>|>.*?</\1\s*>)", re.S | re.I)


def wiki_template(match):
    # リンクの中の | では区切らない（{{要出典範囲|[[東京都|東京]]は…}} が割れないように。elvezjp/ixv-lm#8）
    body = re.sub(r"\[\[[^\[\]]*\]\]", lambda m: m.group().replace("|", "\x00"), match.group(1))
    parts = [p.strip().replace("\x00", "|") for p in body.split("|")]
    name = parts[0].strip().lower().replace(" ", "_")
    args = [p for p in parts[1:] if "=" not in p.split("[[")[0]]
    if name in WIKI_KEEP_FIRST and args:
        return args[0]
    if name in WIKI_KEEP_LAST and args:
        return args[-1]
    return ""


def wiki_link(match):
    target, _, label = match.group(1).partition("|")
    if WIKI_DROP_LINK.match(target.strip()) or target.strip().startswith(":category"):
        return ""
    return (label or target).split("|")[-1] if label else target.split("#")[0] or target


def wiki_text(raw):
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    text = WIKI_TAG_DROP.sub("", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    for _ in range(10):  # 入れ子は内側から
        new = re.sub(r"\{\{([^{}]*)\}\}", wiki_template, text)
        new = re.sub(r"\{\|[^{}]*?\|\}", "", new, flags=re.S)
        new = re.sub(r"\[\[([^\[\]]*)\]\]", wiki_link, new)
        if new == text:
            break
        text = new
    text = re.sub(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^[*#:;]+\s*", " ", text, flags=re.M)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = html.unescape(text)
    text = re.sub(r"^\s*[|!{}].*$", "", text, flags=re.M)  # 落としきれなかった表の行
    return text


def docs_jawiki_dump(stats):
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    proc = subprocess.Popen(["bzcat", str(JAWIKI_DUMP)], stdout=subprocess.PIPE)
    for _, elem in ET.iterparse(proc.stdout, events=("end",)):
        if not elem.tag.endswith("}page"):
            continue
        tag = elem.tag[: -len("page")]
        if elem.findtext(tag + "ns") == "0" and elem.find(tag + "redirect") is None:
            title = elem.findtext(tag + "title") or ""
            raw = elem.findtext(f"{tag}revision/{tag}text") or ""
            if raw.lstrip().lower().startswith(("#redirect", "#転送")):
                stats["redirect"] = stats.get("redirect", 0) + 1
            else:
                yield elem.findtext(tag + "id"), f"{title}\n\n{wiki_text(raw)}"
        elem.clear()
    proc.wait()


# エルブズの文章 ---------------------------------------------------------

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_text(raw):
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(raw)).read("word/document.xml"))
    return "\n".join("".join(t.text or "" for t in p.iter(WORD + "t")) for p in root.iter(WORD + "p"))


TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".py", ".ts", ".js", ".sh", ".toml", ".csv"}
SKIP_NAMES = {"source.json", "SHA256SUMS", "LICENSE", ".DS_Store"}


def docs_dir(base):
    def docs(stats):
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.name in SKIP_NAMES or any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            doc = str(path.relative_to(base))
            if path.suffix == ".docx":
                yield doc, docx_text(path.read_bytes())
            elif path.suffix in TEXT_SUFFIXES:
                yield doc, path.read_text(encoding="utf-8")
            else:
                stats.setdefault("skipped_files", []).append(doc)
    return docs


# 論文紹介と OSS 紹介は LLM で書いた記事なので外す（2026-11-08 決定。最後の記事に LLM で書いたとある）
BLOG_LLM_TITLE = re.compile(r"(論文紹介|OSS紹介)")


def docs_hdkworks(stats):
    posts = json.loads((RAW / "hdkworks-blog/posts-20261108.json").read_text(encoding="utf-8"))["posts"]
    stats["llm_written"] = 0
    for post in posts:
        if BLOG_LLM_TITLE.search(post["title"]):
            stats["llm_written"] += 1
            continue
        if len(post["text"].strip()) < MIN_CHARS:  # 画像だけの記事などは題名だけ残るので外す
            stats["short"] += 1
            continue
        yield str(post["id"]), f"{post['title']}\n\n{post['text']}"


SOURCES = {
    "aozora": docs_aozora,
    "hdkworks-blog": docs_hdkworks,
    "egov-law": docs_egov,
    "wikipedia-ja": docs_wikipedia,
    "wikipedia-ja-20261101": docs_jawiki_dump,
    "elvez-books": docs_dir(RAW / "elvez-books/spec-driven-dev-guide"),
    "elvez-sdd-repo": docs_dir(RAW / "elvez-books/SDD"),
    "elvez-screen-driven": docs_dir(RAW / "elvez-books/sdd-guide-2024"),
}


# 大きいデータ: 入力ファイルごとに並列で加工する ---------------------------

FINEWEB2_MIN_SCORE = 3.0  # 2026-11-08 決定。約430億トークンで ja-edu の枠 280億を満たす


def fineweb2_file(path):
    table = pq.read_table(path, columns=["id", "text", "score"])
    for doc, text, score in zip(*(table.column(c).to_pylist() for c in ("id", "text", "score"))):
        if score >= FINEWEB2_MIN_SCORE:
            yield doc, text


FINEPDFS_MIN_EDU = 1.2  # 2026-11-08 決定。ページごとの教育スコアの平均。約95億トークンで ja-pdf の枠 80億を満たす


def finepdfs_file(path):
    # 1ファイルが約3GBあるので、少しずつ読んでメモリを抑える
    columns = ["id", "text", "fw_edu_scores"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_000, columns=columns):
        for doc, text, scores in zip(*(batch.column(c).to_pylist() for c in columns)):
            if scores and sum(scores) / len(scores) >= FINEPDFS_MIN_EDU:
                yield doc, text


PARALLEL = {
    "fineweb2-edu-ja": (lambda: sorted((RAW / "fineweb-2-edu-japanese/data").glob("*.parquet")), fineweb2_file),
    "finepdfs-ja": (lambda: sorted((RAW / "finepdfs-ja/data/jpn_Jpan/train").glob("*.parquet")), finepdfs_file),
}


def clean_rows(pairs, stats, seen):
    rows = []
    for doc, text in pairs:
        stats["docs_in"] += 1
        text = normalize(text)
        if len(text) < MIN_CHARS:
            stats["short"] += 1
            continue
        digest = hashlib.sha256(text.encode()).digest()
        if digest in seen:
            stats["duplicate"] += 1
            continue
        seen.add(digest)
        rows.append((doc, text))
        stats["docs_out"] += 1
        stats["chars"] += len(text)
        stats["bytes"] += len(text.encode())
    return rows


def parallel_part(job):
    # 重複除去はファイルの中だけ。元のデータは MinHash で重複除去済み
    source, index, path = job
    part = OUT / source / f"part-{index:05d}.parquet"
    done = part.with_suffix(".json")
    if part.exists() and done.exists():
        return json.loads(done.read_text())  # 前回の続き。part と json は書き終わってから置く
    stats = {"docs_in": 0, "docs_out": 0, "short": 0, "duplicate": 0, "chars": 0, "bytes": 0}
    rows = clean_rows(PARALLEL[source][1](path), stats, set())
    table = pa.table({"doc": [r[0] for r in rows], "text": [r[1] for r in rows]})
    tmp = OUT / source / f".part-{index:05d}.tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, part)
    done.write_text(json.dumps({**stats, "input": path.name}))
    return stats


def run_parallel(source, dest, workers, limit):
    files = PARALLEL[source][0]()[:limit]
    stats = {"docs_in": 0, "docs_out": 0, "short": 0, "duplicate": 0, "chars": 0, "bytes": 0}
    jobs = [(source, i, path) for i, path in enumerate(files)]
    with ProcessPoolExecutor(workers) as pool:
        for done, part in enumerate(pool.map(parallel_part, jobs, chunksize=1), 1):
            for key in stats:
                stats[key] += part[key]
            if done % 50 == 0 or done == len(jobs):
                print(f"{datetime.datetime.now():%H:%M:%S} {done}/{len(jobs)} {stats['docs_out']} 文書", flush=True)
    stats["inputs"] = len(files)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=sorted(SOURCES) + sorted(PARALLEL))
    parser.add_argument("--workers", type=int, default=4, help="並列の数。学習マシンでは GPU の処理と重ねない")
    parser.add_argument("--limit", type=int, help="先頭のこの数の入力だけ処理する（メモリの見積もり用）")
    args = parser.parse_args()

    dest = OUT / args.source
    dest.mkdir(parents=True, exist_ok=True)
    if args.source in PARALLEL:
        # 途中で止まっても、書き終えた part から続ける
        stats = run_parallel(args.source, dest, args.workers, args.limit)
    else:
        for old in dest.glob("part-*.parquet"):
            old.unlink()
        stats = {"docs_in": 0, "docs_out": 0, "short": 0, "duplicate": 0, "chars": 0, "bytes": 0}
        rows = clean_rows(SOURCES[args.source](stats), stats, set())
        for part, start in enumerate(range(0, len(rows), PART_DOCS)):
            chunk = rows[start : start + PART_DOCS]
            table = pa.table({"doc": [r[0] for r in chunk], "text": [r[1] for r in chunk]})
            pq.write_table(table, dest / f"part-{part:05d}.parquet", compression="zstd")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    (dest / "source.json").write_text(json.dumps(
        {"source": args.source, "made": f"{datetime.date.today()}", "script_commit": commit,
         "min_chars": MIN_CHARS, **stats}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "skipped_files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
