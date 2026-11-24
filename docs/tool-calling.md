# ツール呼び出しの学習（案）

2026-11-12 時点の案。まだ決めていない。決めたら、決めた日付を書いて「案」を外す。

## ねらい

IXV のハーネスから、IXV-LM が決まったツールを正しく呼べるようにする。自律的にコードを書くことは狙わない（[README](../README_ja.md) の「できるようにすること」）。

## IXV が使っているツール

IXV のハーネスのツール定義にある。OpenAI 互換のツール呼び出し形式で送られる。

| ツール | 引数 |
|---|---|
| read | path、offset、limit |
| ls | path、limit |
| find | pattern（glob）、path、limit |
| grep | pattern、path、glob、ignoreCase、literal、context、limit |
| write | path、content |
| edit | path と置き換え |
| bash | command |

いまのハーネスは、ツールの説明などで約 5,600 トークンを送る。0.3B には重い。

## 書式

トークナイザーには、ツール用の特別なトークンを最初から予約してある。

```
<|system|>{ツールの一覧}
<|user|>README を要約して
<|assistant|><|tool_call|>{"name": "read", "arguments": {"path": "README.md"}}<|/tool_call|>
<|tool_result|>{ファイルの中身}<|/tool_result|>
<|assistant|>{要約}<|endoftext|>
```

（見やすさのために改行しているが、実際は特別なトークンのあいだに改行を入れない。特別なトークンは文字から作らず、番号を直接入れる。）
呼び出しは `<|/tool_call|>` で生成を止め、ハーネスが結果を `<|tool_result|>` で返して、もう一度 `<|assistant|>` から書かせる。

学習の対象は、アシスタントの部分（呼び出しと答え）だけにする。会話だけの追加学習も、同じ書式に揃える（いまの `ユーザー：…` からの切り替え）。

## データの作り方

1. **プログラムで組み立てる（中心）**。読むだけのツール（read、ls、find、grep）から始める。実在のファイルとリポジトリ（台帳で学習してよいもの）に、実際にツールを走らせて結果を得る。依頼の文と呼び出しは、型をもとにプログラムで作る。LLM には書かせない
2. **公開データ**。ライセンスと作り方（どの LLM で作ったか）を確かめ、台帳に載せてから使う。Claude が作ったものは入れない
3. **人が書くお手本**。判断が難しい場面だけ、少し

## 小さいモデルに合わせること

- ツールは数個、説明は短く。IXV-LM のときは、ハーネスが短い一覧を送るようにする（ハーネスの変更。いつやるかは未定）
- ファイルの中身を受けるので、2,048 トークンより長い文脈の学習が要る
- 評価は自動で数える。呼び出しの形（JSON として正しいか）、ツール名、引数が合っているか

## 最初の一歩

read、ls、grep の3つで、組み立てたデータを数千件作り、リリースの日の会話の追加学習に混ぜて、0.3B でどこまで呼べるかを見る。

- 組み立ては `corpus/make_tool_sft.py`。台帳で承認済みの文章（elvez-sdd-repo、hdkworks-blog、wikipedia-ja）を仮のフォルダに並べ、ツールを実際に走らせる。結果の形は IXV のツールに合わせた（read は本文、ls は名前順で末尾に `/`、grep は `パス:行: 本文`、なければ `No matches found`）
- 作業は6種類（ls で数える、ls で並べる、read で先頭の行、read で行を指定、grep で探す、grep で見つからない）
- ユーザーの依頼と答えの文の型は `corpus/tool_sft_templates.toml` に人が書く（このリポジトリには入っていない。公開していないので、自分で書く）。書き方は [tool-templates-guide.md](tool-templates-guide.md)。型がないとデータは作らない。`--preview` で、型なしで形だけ確かめられる
- 会話は OpenAI 形式の messages で出す。学習の書式への変換は `train/chat_format.py`（2026-11-13）
- `sft.py --chat-format tokens`（いまの既定）で上の書式になる。r1 までの `ユーザー：…` は `--chat-format plain`。書式は run の config.json に残り、`chat.py` はそれに合わせる
- 試すときは `chat.py --tools フォルダ` で、そのフォルダの中だけで read、ls、grep を実際に走らせる（読むだけ）

作り方と混ぜ方（リリースの追加学習）:

```bash
.venv/bin/python corpus/make_tool_sft.py --count 3000 --out ~/ixv-lm-data/sft/tool-v1.jsonl   # r2
.venv/bin/python corpus/make_tool_sft.py --count 6000 --seed 2 --out ~/ixv-lm-data/sft/tool-v2.jsonl   # r3 から（作り方を直した版。v1 はもう同じには作れない）
.venv/bin/python corpus/make_tool_sft.py --count 900 --seed 3 --keep-files --out ~/ixv-lm-data/sft/tool-eval-v2.jsonl   # 採点用（通しの採点のため、仮のフォルダも残す）
# sft.py に --sft ~/ixv-lm-data/sft/tool-v1.jsonl を足す（会話のデータと一緒に）
```

3000 件で約 20 万トークン（答えの側）、長いもので 1,200 トークン弱。2,048 に収まる。
