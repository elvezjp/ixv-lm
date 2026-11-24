"""学習データ台帳を検査する。誤りがあれば終了コード 1 で止まる。"""
import sys
import tomllib
from collections import Counter
from pathlib import Path

REQUIRED = ("id", "name", "slot", "class", "status", "url", "version", "acquired", "license", "extra_terms",
            "redistribution", "pii", "filters", "helper_models", "used_tokens_b", "used_checkpoints",
            "approver", "approved", "notes")
STATUSES = ("候補", "実験のみ", "承認済み", "使用中")
# 本番に入れる前に埋まっていなければならない項目
BEFORE_APPROVAL = ("url", "version", "acquired", "license", "redistribution", "pii", "filters")


def check(ledger):
    errors = []
    budget = ledger["budget"]
    sources = ledger["source"]
    if sum(budget.values()) != ledger["meta"]["total_target_b"]:
        errors.append(f"配分の合計 {sum(budget.values())} が目標 {ledger['meta']['total_target_b']} と違う")
    for name, count in Counter(s.get("id") for s in sources).items():
        if count > 1:
            errors.append(f"{name}: id が重複している")
    for s in sources:
        label = s.get("id", "?")
        missing = [k for k in REQUIRED if k not in s]
        if missing:
            errors.append(f"{label}: 項目がない {missing}")
            continue
        if s["class"] not in ("A", "B"):
            errors.append(f"{label}: class は A か B（使わないものは excluded に書く）")
        if s["status"] not in STATUSES:
            errors.append(f"{label}: status が不正 {s['status']}")
        if s["slot"] not in budget:
            errors.append(f"{label}: 配分にない slot {s['slot']}")
        if s["status"] in ("承認済み", "使用中"):
            if not (s["approver"] and s["approved"]):
                errors.append(f"{label}: 承認者と承認日がない")
            empty = [k for k in BEFORE_APPROVAL if not s[k]]
            if empty:
                errors.append(f"{label}: 承認前に埋める項目が空 {empty}")
        elif s["used_tokens_b"] or s["used_checkpoints"]:
            errors.append(f"{label}: 未承認なのに使用の記録がある")
    for slot in sorted(set(budget) - {s.get("slot") for s in sources}):
        errors.append(f"{slot}: 配分に対応するデータがない")
    return errors


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("ledger.toml")
    ledger = tomllib.loads(path.read_text())
    errors = check(ledger)
    sources = ledger["source"]
    print(f"データ {len(sources)} 件、除外 {len(ledger['excluded'])} 件、配分 {sum(ledger['budget'].values())}B")
    print("分類:", dict(sorted(Counter(s.get("class") for s in sources).items())))
    print("状態:", dict(Counter(s.get("status") for s in sources)))
    unknown = [s["id"] for s in sources if not s.get("license")]
    print(f"ライセンス未確認 {len(unknown)} 件:", ", ".join(unknown))
    for error in errors:
        print("誤り:", error)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
