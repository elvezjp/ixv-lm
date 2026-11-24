"""会話をトークン列にする書式。追加学習とチャットで同じものを使う。

書式は2つ。追加学習の run の config.json の args.chat_format に残し、チャットはそれに合わせる。

plain（r1 まで）:
    ユーザー：{質問}
    アシスタント：{答え}<|endoftext|>

tokens（予約しておいた特別なトークン。ツール呼び出しもこちら）:
    <|system|>{ツールの一覧}<|user|>{依頼}<|assistant|><|tool_call|>{"name": …, "arguments": …}<|/tool_call|>
    <|tool_result|>{結果}<|/tool_result|><|assistant|>{答え}<|endoftext|>
    特別なトークンは文字から作らず、番号をそのまま入れる（区切りに改行は入れない）。
    呼び出しは <|/tool_call|> で止まり、ハーネスが結果を <|tool_result|> で返す。

学習の対象は、アシスタントが書く部分（答え、呼び出し、終わりの印）だけ。
"""
import json

USER = "ユーザー："
ASSISTANT = "\nアシスタント："
FORMATS = ("plain", "tokens")
SPECIAL = ("<|system|>", "<|user|>", "<|assistant|>", "<|tool_call|>", "<|/tool_call|>",
           "<|tool_result|>", "<|/tool_result|>")


def special_ids(tokenizer):
    ids = {name: tokenizer.token_to_id(name) for name in SPECIAL}
    missing = [name for name, i in ids.items() if i is None]
    if missing:
        raise SystemExit(f"トークナイザーに特別なトークンがありません: {missing}")
    return ids


def call_text(call):
    return json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)


def encode_plain(tokenizer, messages, eot):
    ids, mask = [], []
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] == "user":
            piece = tokenizer.encode(USER + message["content"] + ASSISTANT).ids
            ids += piece
            mask += [0] * len(piece)
        elif message["role"] == "assistant":
            if message.get("tool_calls"):
                raise ValueError("plain の書式ではツール呼び出しを書けません")
            piece = tokenizer.encode(message["content"]).ids + [eot]
            ids += piece
            mask += [1] * len(piece)
        else:
            raise ValueError(f"plain の書式で使えない役割: {message['role']}")
    return ids, mask


def encode_tokens(tokenizer, messages, eot, open_assistant=False):
    """open_assistant=True のとき、最後に <|assistant|> を置いて答えを書き始める直前で止める。"""
    sp = special_ids(tokenizer)
    ids, mask = [], []

    def add(piece, target):
        ids.extend(piece)
        mask.extend([target] * len(piece))

    def text(s):
        return tokenizer.encode(s).ids

    for message in messages:
        role = message["role"]
        if role == "system":
            add([sp["<|system|>"]] + text(message["content"]), 0)
        elif role == "user":
            add([sp["<|user|>"]] + text(message["content"]), 0)
        elif role == "assistant":
            add([sp["<|assistant|>"]], 0)
            calls = message.get("tool_calls") or []
            if message.get("content"):
                add(text(message["content"]), 1)
            for call in calls:
                add([sp["<|tool_call|>"]] + text(call_text(call)) + [sp["<|/tool_call|>"]], 1)
            if not calls:
                add([eot], 1)
        elif role == "tool":
            add([sp["<|tool_result|>"]] + text(message["content"]) + [sp["<|/tool_result|>"]], 0)
        else:
            raise ValueError(f"使えない役割: {role}")
    if open_assistant:
        add([sp["<|assistant|>"]], 0)
    return ids, mask


def encode_messages(tokenizer, messages, eot, chat_format="plain"):
    """トークン列と、学習の対象かどうかの印（1 がアシスタントの書く部分）を返す。"""
    if chat_format == "tokens":
        return encode_tokens(tokenizer, messages, eot)
    return encode_plain(tokenizer, messages, eot)


def prompt_ids(tokenizer, messages, eot, chat_format="plain"):
    """最後がユーザーの発言かツールの結果で終わる会話から、アシスタントが書き始める直前までを作る。"""
    if chat_format == "tokens":
        ids, _ = encode_tokens(tokenizer, messages, eot, open_assistant=True)
        return ids
    ids, _ = encode_plain(tokenizer, messages, eot)
    return ids


def stop_ids(tokenizer, eot, chat_format="plain"):
    """生成を止める印。tokens ではツール呼び出しの終わりでも止める。"""
    if chat_format == "tokens":
        return {eot, special_ids(tokenizer)["<|/tool_call|>"]}
    return {eot}
