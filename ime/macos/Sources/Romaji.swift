// ローマ字をひらがなにする。打った途中のローマ字（"k" や "ky"）は残す。
enum Romaji {
    static let table: [String: String] = {
        var t: [String: String] = [
            "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
            "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
            "sa": "さ", "si": "し", "shi": "し", "su": "す", "se": "せ", "so": "そ",
            "ta": "た", "ti": "ち", "chi": "ち", "tu": "つ", "tsu": "つ", "te": "て", "to": "と",
            "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
            "ha": "は", "hi": "ひ", "hu": "ふ", "fu": "ふ", "he": "へ", "ho": "ほ",
            "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
            "ya": "や", "yu": "ゆ", "yo": "よ",
            "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
            "wa": "わ", "wo": "を", "nn": "ん", "n'": "ん", "xn": "ん",
            "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
            "za": "ざ", "zi": "じ", "ji": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
            "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",
            "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
            "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
            "va": "ゔぁ", "vi": "ゔぃ", "vu": "ゔ", "ve": "ゔぇ", "vo": "ゔぉ",
            "fa": "ふぁ", "fi": "ふぃ", "fe": "ふぇ", "fo": "ふぉ",
            "sha": "しゃ", "shu": "しゅ", "she": "しぇ", "sho": "しょ",
            "cha": "ちゃ", "chu": "ちゅ", "che": "ちぇ", "cho": "ちょ",
            "ja": "じゃ", "ju": "じゅ", "je": "じぇ", "jo": "じょ",
            "tsa": "つぁ", "thi": "てぃ", "dhi": "でぃ", "twu": "とぅ", "dwu": "どぅ",
            "wi": "うぃ", "we": "うぇ", "ye": "いぇ",
            "xa": "ぁ", "xi": "ぃ", "xu": "ぅ", "xe": "ぇ", "xo": "ぉ",
            "la": "ぁ", "li": "ぃ", "lu": "ぅ", "le": "ぇ", "lo": "ぉ",
            "xya": "ゃ", "xyu": "ゅ", "xyo": "ょ", "lya": "ゃ", "lyu": "ゅ", "lyo": "ょ",
            "xtu": "っ", "ltu": "っ", "xtsu": "っ", "xwa": "ゎ",
            "-": "ー", ",": "、", ".": "。", "[": "「", "]": "」", "/": "・",
            "?": "？", "!": "！", "~": "〜", "(": "（", ")": "）",
        ]
        let rows: [(String, String)] = [("k", "き"), ("s", "し"), ("t", "ち"), ("n", "に"), ("h", "ひ"), ("m", "み"),
                                         ("r", "り"), ("g", "ぎ"), ("z", "じ"), ("d", "ぢ"), ("b", "び"), ("p", "ぴ"),
                                         ("j", "じ"), ("c", "ち")]
        for (c, base) in rows {
            t[c + "ya"] = base + "ゃ"
            t[c + "yu"] = base + "ゅ"
            t[c + "ye"] = base + "ぇ"
            t[c + "yo"] = base + "ょ"
        }
        return t
    }()

    static let prefixes: Set<String> = {
        var p = Set<String>()
        for key in table.keys {
            for i in 1..<key.count { p.insert(String(key.prefix(i))) }
        }
        return p
    }()

    /// 未確定のローマ字 pending に 1 字足し、確定したかなと、残りのローマ字を返す。
    static func feed(_ pending: String, _ char: Character) -> (kana: String, pending: String) {
        let s = pending + String(char).lowercased()
        if let k = table[s] { return (k, "") }
        if prefixes.contains(s) { return ("", s) }
        // 促音（kk → っk）と、子音の前の n（nk → んk）
        if s.count >= 2 {
            let first = s.first!, rest = String(s.dropFirst())
            if first == "n" && !"aiueony'".contains(rest.first!) {
                let (k, p) = feed("", rest.first!)
                return ("ん" + k, p)
            }
            if first == rest.first! && first.isLetter && !"aiueon".contains(first) {
                return ("っ", rest)
            }
            // どれにも当たらない：前のローマ字をそのまま出して、打った字からやり直す
            let (k, p) = feed("", rest.last!)
            return (String(s.dropLast()) + k, p)
        }
        return (s, "")
    }

    /// 打ち終わりで残ったローマ字を片付ける（n だけなら ん）。
    static func flush(_ pending: String) -> String {
        pending == "n" ? "ん" : pending
    }

    static func katakana(_ hira: String) -> String {
        String(String.UnicodeScalarView(hira.unicodeScalars.map {
            (0x3041...0x3096).contains($0.value) ? Unicode.Scalar($0.value + 0x60)! : $0
        }))
    }
}
