import Cocoa
import InputMethodKit

// 打ち方（ime/README.md にも書いた）
// - ローマ字でかなを打つ。数字はそのまま、Shift を押した英字は英字のまま入る。
// - スペースで変換。文節ごとに候補が出る。スペースか ↓ で次の候補、↑ で前の候補。
// - ← → で文節を移る。Shift+← → で文節を縮める・伸ばす（後ろは変換し直す）。
// - 数字の 1〜9 で候補を選び、次の文節へ（最後の文節なら確定）。Enter で全部を確定。
// - Esc で変換をやめてかなに戻る（もう一度で消す）。削除でも変換をやめてかなに戻る。
// - F6 ひらがな、F7 カタカナ、F8 半角カタカナ、F9 全角英字、F10 半角英字。
// - 英数キーで直接入力、かなキーで日本語入力。
// - 打っている途中は予測変換を出す。Tab で選び（もう一度で次）、Enter で確定。
// - 設定は、入力メニュー（メニューバーの「漢」）の「設定…」から（Settings.swift）。
// 確定した変換のうち、使う人が候補を選び直した文節だけを ime/server.py に覚えさせる（学習、elvezjp/ixv-lm#11）。
// 予測変換とライブ変換は、メインスレッドで待たずに問い合わせ、返ってきたら表示を直す。古い返事は捨てる（elvezjp/ixv-lm#7）。
// スペースで変換したら、まず検算なしの結果（ライブ変換の返事があればそれ）をすぐ出し、
// 検算ありの返事が来たら差し替える。それまでに候補を選び直していれば、返事は捨てる（elvezjp/ixv-lm#2）。
@objc(IXVInputController)
class IXVInputController: IMKInputController {
    private struct Segment {
        var kana: String
        var candidates: [String]
        var index = 0
        var chosen = false  // 使う人が候補を選び直したか（選び直したものだけを覚える）
        var text: String { candidates.isEmpty ? kana : candidates[index] }
    }

    private var kana = ""            // 打ったかな（ひらがな、数字、英字）
    private var pending = ""         // まだかなにならないローマ字
    private var typed = ""           // 打ったキーそのまま（F9、F10 用）
    private var segments: [Segment] = []
    private var focus = 0
    private var direct = false       // 英数キーで直接入力
    private var predictions: [String] = []
    private var predIndex = -1       // Tab で選んでいる予測の番号
    private var live = ""            // ライブ変換の表示
    private var liveReply: [Server.Reply] = []  // ライブ変換の返事（検算なし）と、その読み
    private var liveKana = ""
    private var refining = 0         // 検算ありの返事を待っている変換の番号（0 は待っていない）
    private var conversions = 0      // スペースで変換するたびに増やす
    private var generation = 0       // 打つたびに増やす。古い問い合わせの返事を捨てるのに使う
    private var context: String { Settings.useContext ? Self.context : "" }

    // 入力メニュー（メニューバーの「漢」）に足す項目
    override func menu() -> NSMenu! {
        let menu = NSMenu()
        for (title, action) in [("設定…", #selector(openSettings(_:))),
                                ("ユーザー辞書を編集…", #selector(openUserDictionary(_:)))] {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        return menu
    }

    @objc func openSettings(_ sender: Any?) {
        NSLog("IXVKanji: openSettings")
        DispatchQueue.main.async { SettingsWindow.shared.show() }
    }

    override func showPreferences(_ sender: Any!) { openSettings(sender) }

    @objc func openUserDictionary(_ sender: Any?) {
        let file = Settings.storeFolder.appendingPathComponent("user_dict.tsv")
        if !FileManager.default.fileExists(atPath: file.path) {
            try? FileManager.default.createDirectory(at: Settings.storeFolder, withIntermediateDirectories: true)
            try? "# 読み（ひらがなかカタカナ）<TAB>変換。# で始まる行は飛ばす。UTF-8 で保存する\n".write(to: file, atomically: true, encoding: .utf8)
        }
        NSWorkspace.shared.open([file], withApplicationAt: URL(fileURLWithPath: "/System/Applications/TextEdit.app"),
                                configuration: NSWorkspace.OpenConfiguration())
    }
    private static var context = ""  // 確定した文（左の文脈として渡す）

    private var composing: Bool { !kana.isEmpty || !pending.isEmpty }
    private var converting: Bool { !segments.isEmpty }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event = event, event.type == .keyDown, let client = sender as? IMKTextInput else { return false }
        switch event.keyCode {
        case 102: // 英数
            if composing { commit(client) }
            direct = true
            return true
        case 104: // かな
            direct = false
            return true
        default:
            break
        }
        if direct { return false }
        let flags = event.modifierFlags.intersection([.command, .control, .option, .shift])
        if !flags.subtracting(.shift).isEmpty {
            if composing { commit(client) }
            return false
        }
        let shift = flags.contains(.shift)

        if converting {
            switch event.keyCode {
            case 36, 76: commit(client); return true                       // Enter
            case 49, 125: choose(segments[focus].index + 1, client); return true // スペース、↓
            case 126: choose(segments[focus].index - 1, client); return true     // ↑
            case 123: if shift { resize(-1, client) } else { moveFocus(-1, client) }; return true // ←
            case 124: if shift { resize(1, client) } else { moveFocus(1, client) }; return true   // →
            case 53, 51: backToKana(client); return true                    // Esc、削除
            case 97, 98, 100, 101, 109: special(event.keyCode, client); return true
            default: break
            }
            let pageStart = segments[focus].index / Settings.candidateCount * Settings.candidateCount
            if let ch = event.characters?.first, let n = ch.wholeNumberValue, n >= 1,
               n <= min(9, Settings.candidateCount), pageStart + n - 1 < segments[focus].candidates.count {
                choose(pageStart + n - 1, client)
                if focus == segments.count - 1 { commit(client) } else { moveFocus(1, client) }
                return true
            }
        } else if composing {
            switch event.keyCode {
            case 36, 76: commit(client); return true
            case 49: convert(client); return true
            case 51:
                if !pending.isEmpty { pending.removeLast() } else { kana.removeLast() }
                if !typed.isEmpty { typed.removeLast() }
                show(client)
                return true
            case 53: clear(client); return true
            case 97, 98, 100, 101, 109: special(event.keyCode, client); return true
            case 48: // Tab：予測を選ぶ
                if !predictions.isEmpty {
                    predIndex = (predIndex + 1) % predictions.count
                    show(client, refresh: false)
                }
                return true
            case 123, 124, 125, 126: return true
            default: break
            }
        } else if event.keyCode == 36 || event.keyCode == 76 {
            Self.context = ""
            return false
        } else if event.keyCode == 49 && Settings.fullWidthSpace && !shift {
            client.insertText("　", replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
            return true
        }

        guard let chars = event.characters, chars.count == 1, let ch = chars.first, ch.isASCII,
              !ch.isWhitespace, ch.asciiValue! >= 0x21 else {
            if composing { commit(client) }
            return false
        }
        if converting { commit(client) }
        typed.append(ch)
        if ch.isNumber || (shift && ch.isLetter) {
            let raw = String(ch)
            kana += Romaji.flush(pending) + (Settings.fullWidthAlnum ? Variants.fullWidth(raw) : raw)
            pending = ""
        } else if ch.isLetter || Romaji.table[String(ch)] != nil {
            let (k, p) = Romaji.feed(pending, ch)
            kana += Settings.punctuate(k)
            pending = p
        } else {
            kana += Romaji.flush(pending) + String(ch)
            pending = ""
        }
        show(client)
        return true
    }

    // ---------- 表示 ----------
    private func show(_ client: IMKTextInput, refresh: Bool = true) {
        let marked = NSMutableAttributedString()
        if !converting && refresh {
            predIndex = -1
            predictions = []
            live = ""
            liveReply = []
            generation += 1
            let text = kana + Romaji.flush(pending), g = generation
            // 返事が来たときに、まだ同じ打ちかけなら表示を直す
            let current = { [weak self] in self.map { $0.generation == g && !$0.converting } ?? false }
            if Settings.prediction && text.count >= 2 && pending.isEmpty {
                Server.predict(kana: text) { [weak self] list in
                    guard let self = self, current() else { return }
                    self.predictions = list
                    self.show(client, refresh: false)
                }
            }
            if Settings.liveConversion && text.count >= 2 && pending.isEmpty {
                Server.convert(context: context, kana: text, segments: nil, fast: true) { [weak self] reply in
                    guard let self = self, current(), self.predIndex < 0 else { return }
                    self.live = reply.map { $0.candidates.first ?? $0.kana }.joined()
                    self.liveReply = reply
                    self.liveKana = text
                    self.show(client, refresh: false)
                }
            }
        }
        if !converting && predIndex >= 0 {
            marked.append(NSAttributedString(string: predictions[predIndex],
                                             attributes: [.underlineStyle: NSUnderlineStyle.thick.rawValue]))
        } else if !converting && !live.isEmpty {
            marked.append(NSAttributedString(string: live + pending,
                                             attributes: [.underlineStyle: NSUnderlineStyle.single.rawValue]))
        } else if converting {
            for (i, seg) in segments.enumerated() {
                let style = i == focus ? NSUnderlineStyle.thick : NSUnderlineStyle.single
                marked.append(NSAttributedString(string: seg.text, attributes: [.underlineStyle: style.rawValue]))
            }
        } else {
            marked.append(NSAttributedString(string: kana + pending,
                                             attributes: [.underlineStyle: NSUnderlineStyle.single.rawValue]))
        }
        client.setMarkedText(marked, selectionRange: NSRange(location: marked.length, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        if converting && segments[focus].candidates.count > 1 {
            CandidatesPanel.shared.show(segments[focus].candidates, selected: segments[focus].index, client: client)
        } else if !converting && !predictions.isEmpty {
            CandidatesPanel.shared.show(predictions, selected: predIndex, client: client, numbered: false)
        } else {
            CandidatesPanel.shared.hide()
        }
    }

    // ---------- 変換 ----------
    private func convert(_ client: IMKTextInput) {
        kana += Romaji.flush(pending)
        pending = ""
        // まず検算なしの結果をすぐ出す。ライブ変換の返事が同じ読みなら、それを使う
        let quick = !liveReply.isEmpty && liveKana == kana
            ? liveReply.map { Segment(kana: $0.kana, candidates: $0.candidates) }
            : request(kana: kana, lengths: nil, extraContext: "", fast: true)
        segments = quick
        focus = 0
        conversions += 1
        refining = 0
        show(client)
        guard Settings.useModel else { return }
        // 検算ありの変換を待たずに問い合わせ、返事が来たら、まだ選び直していなければ差し替える
        let n = conversions, reading = kana
        refining = n
        Server.convert(context: context, kana: reading, segments: nil, fast: false, timeout: 8) { [weak self] reply in
            guard let self = self, self.refining == n, self.converting, self.kana == reading else { return }
            self.refining = 0
            guard !reply.isEmpty else { return }
            self.segments = reply.map { Segment(kana: $0.kana, candidates: $0.candidates) }
            self.focus = min(self.focus, self.segments.count - 1)
            self.show(client)
        }
    }

    private func request(kana: String, lengths: [Int]?, extraContext: String, fast: Bool? = nil) -> [Segment] {
        let reply = Server.convert(context: context + extraContext, kana: kana, segments: lengths,
                                   fast: fast ?? !Settings.useModel)
        if reply.isEmpty {
            return [Segment(kana: kana, candidates: Variants.basic(kana))]
        }
        return reply.map { Segment(kana: $0.kana, candidates: $0.candidates) }
    }

    private func choose(_ i: Int, _ client: IMKTextInput) {
        let count = segments[focus].candidates.count
        guard count > 0 else { return }
        segments[focus].index = (i + count) % count
        segments[focus].chosen = true
        refining = 0  // 選び直したら、検算の返事で差し替えない
        show(client)
    }

    private func moveFocus(_ d: Int, _ client: IMKTextInput) {
        focus = max(0, min(segments.count - 1, focus + d))
        show(client)
    }

    // 注目している文節を d 字ぶん伸ばす・縮める。前の文節はそのまま、後ろは変換し直す。
    private func resize(_ d: Int, _ client: IMKTextInput) {
        let rest = segments[focus...].map { $0.kana }.joined()
        let length = segments[focus].kana.count + d
        guard length >= 1, length <= rest.count else { return }
        refining = 0  // 文節の区切りを変えたら、検算の返事で差し替えない
        let head = String(rest.prefix(length)), tail = String(rest.dropFirst(length))
        let prior = segments[..<focus].map { $0.text }.joined()
        var updated = Array(segments[..<focus])
        updated += request(kana: head, lengths: [head.count], extraContext: prior)
        if !tail.isEmpty {
            updated += request(kana: tail, lengths: nil, extraContext: prior + (updated.last?.text ?? ""))
        }
        segments = updated
        show(client)
    }

    private func backToKana(_ client: IMKTextInput) {
        refining = 0
        segments = []
        focus = 0
        show(client)
    }

    private func special(_ keyCode: UInt16, _ client: IMKTextInput) {
        let all = kana + Romaji.flush(pending)
        pending = ""
        kana = all
        let text: String
        switch keyCode {
        case 97: text = Variants.hiragana(all)                        // F6
        case 98: text = Variants.katakana(all)                        // F7
        case 100: text = Variants.halfKatakana(all)                   // F8
        case 101: text = Variants.fullWidth(typed.isEmpty ? all : typed) // F9
        default: text = typed.isEmpty ? all : typed                   // F10
        }
        refining = 0
        segments = [Segment(kana: all, candidates: [text])]
        focus = 0
        show(client)
    }

    private func clear(_ client: IMKTextInput) {
        predictions = []
        live = ""
        kana = ""
        pending = ""
        typed = ""
        segments = []
        refining = 0
        show(client)
    }

    private func commit(_ client: IMKTextInput) {
        let text: String
        if converting {
            text = segments.map { $0.text }.joined()
            for seg in segments where seg.chosen && seg.candidates.count > 1 && Settings.learning {
                Server.learn(kana: seg.kana, surface: seg.text)
            }
        } else if predIndex >= 0 {
            text = predictions[predIndex]
        } else if !live.isEmpty && pending.isEmpty {
            text = live
        } else {
            text = kana + Romaji.flush(pending)
        }
        client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        Self.context = String((Self.context + text).suffix(40))
        kana = ""
        pending = ""
        typed = ""
        segments = []
        focus = 0
        predictions = []
        predIndex = -1
        live = ""
        liveReply = []
        refining = 0
        generation += 1
        CandidatesPanel.shared.hide()
    }

    override func commitComposition(_ sender: Any!) {
        if composing || converting, let client = sender as? IMKTextInput { commit(client) }
    }

    override func deactivateServer(_ sender: Any!) {
        commitComposition(sender)
        CandidatesPanel.shared.hide()
        super.deactivateServer(sender)
    }
}

// ひらがな・カタカナ・半角・全角の書き換え
enum Variants {
    static func hiragana(_ s: String) -> String {
        s.applyingTransform(.hiraganaToKatakana, reverse: true) ?? s
    }
    static func katakana(_ s: String) -> String {
        s.applyingTransform(.hiraganaToKatakana, reverse: false) ?? s
    }
    static func halfKatakana(_ s: String) -> String {
        katakana(s).applyingTransform(.fullwidthToHalfwidth, reverse: false) ?? s
    }
    static func fullWidth(_ s: String) -> String {
        s.applyingTransform(.fullwidthToHalfwidth, reverse: true) ?? s
    }
    static func basic(_ s: String) -> [String] {
        Array(NSOrderedSet(array: [s, katakana(s)])) as? [String] ?? [s]
    }
}

// 変換サーバー（ime/server.py）への問い合わせ。つながらなければ空を返す（ひらがなとカタカナだけが候補になる）。
enum Server {
    static let base = "http://127.0.0.1:18190"

    struct Reply {
        let kana: String
        let candidates: [String]
    }

    /// 問い合わせる。done はメインスレッドで呼ぶ。
    private static func send(_ path: String, _ body: [String: Any], timeout: TimeInterval,
                             done: @escaping ([String: Any]?) -> Void) {
        var request = URLRequest(url: URL(string: base + path)!, timeoutInterval: timeout)
        request.httpMethod = "POST"
        var body = body
        body["learning"] = Settings.learning
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        URLSession.shared.dataTask(with: request) { data, _, _ in
            let result = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            DispatchQueue.main.async { done(result) }
        }.resume()
    }

    /// 返事を待つ問い合わせ。スペースで変換したときと設定画面だけで使う（打つたびには使わない）。
    private static func post(_ path: String, _ body: [String: Any]) -> [String: Any]? {
        var request = URLRequest(url: URL(string: base + path)!, timeoutInterval: 8)
        request.httpMethod = "POST"
        var body = body
        body["learning"] = Settings.learning
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        var result: [String: Any]?
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { data, _, _ in
            if let data = data { result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] }
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 9)
        return result
    }

    private static func replies(_ json: [String: Any]?) -> [Reply] {
        guard let segs = json?["segments"] as? [[String: Any]] else { return [] }
        return segs.compactMap { seg in
            guard let k = seg["kana"] as? String, let c = seg["candidates"] as? [String], !c.isEmpty else { return nil }
            return Reply(kana: k, candidates: c)
        }
    }

    private static func convertBody(_ context: String, _ kana: String, _ segments: [Int]?, _ fast: Bool) -> [String: Any] {
        var body: [String: Any] = ["context": context, "kana": kana, "fast": fast]
        if let segments = segments { body["segments"] = segments }
        return body
    }

    static func status() -> [String: Any]? { post("/status", [:]) }

    static func forget() { _ = post("/forget", [:]) }

    /// 予測変換。打つたびに呼ぶので待たない
    static func predict(kana: String, done: @escaping ([String]) -> Void) {
        send("/predict", ["kana": kana, "n": 6], timeout: 2) { json in
            done(json?["candidates"] as? [String] ?? [])
        }
    }

    /// 変換して返事を待つ（スペースで変換したときの検算なしの結果と、文節を伸び縮みしたとき）
    static func convert(context: String, kana: String, segments: [Int]?, fast: Bool = false) -> [Reply] {
        replies(post("/convert", convertBody(context, kana, segments, fast)))
    }

    /// 変換して、返事が来たら done を呼ぶ（ライブ変換と、スペースで変換したあとの検算。待たない）
    static func convert(context: String, kana: String, segments: [Int]?, fast: Bool, timeout: TimeInterval = 2,
                        done: @escaping ([Reply]) -> Void) {
        send("/convert", convertBody(context, kana, segments, fast), timeout: timeout) { done(replies($0)) }
    }

    static func learn(kana: String, surface: String) {
        send("/learn", ["kana": kana, "surface": surface], timeout: 2) { _ in }
    }
}
