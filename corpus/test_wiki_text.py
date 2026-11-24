"""clean.py の wiki_text の回帰テスト。例: .venv/bin/python corpus/test_wiki_text.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean  # noqa: E402

CASES = [
    # テンプレートの中のパイプ付きリンク（elvezjp/ixv-lm#8）
    ("{{要出典範囲|[[東京都|東京]]は日本の首都である|date=2020年1月}}。次の文。", "東京は日本の首都である。次の文。"),
    ("{{lang|en|[[Tokyo|Tokyo City]]}}", "Tokyo City"),
    # リンクの中のテンプレート
    ("[[A|{{lang|en|B}}]]", "B"),
    ("{{仮リンク|山田|en|Yamada}}と[[東京]]", "山田と東京"),
]

failed = 0
for raw, want in CASES:
    got = clean.wiki_text(raw)
    if got != want:
        failed += 1
        print(f"NG {raw!r}\n   want {want!r}\n   got  {got!r}")
print(f"{len(CASES) - failed}/{len(CASES)} ok")
sys.exit(1 if failed else 0)
