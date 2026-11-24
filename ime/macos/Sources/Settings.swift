import Cocoa
import SwiftUI

// 設定。UserDefaults（jp.co.elvez.inputmethod.IXVKanji）に置く。設定画面からも defaults からも変えられる。
enum Settings {
    static let store = UserDefaults.standard
    static let storeFolder = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/IXVKanji")

    static func bool(_ key: String, _ fallback: Bool) -> Bool { store.object(forKey: key) as? Bool ?? fallback }
    static func int(_ key: String, _ fallback: Int) -> Int { store.object(forKey: key) as? Int ?? fallback }

    static var liveConversion: Bool { bool("liveConversion", false) }
    static var prediction: Bool { bool("prediction", true) }
    static var useModel: Bool { bool("useModel", true) }
    static var useContext: Bool { bool("useContext", true) }
    static var learning: Bool { bool("learning", true) }
    static var fullWidthSpace: Bool { bool("fullWidthSpace", false) }
    static var fullWidthAlnum: Bool { bool("fullWidthAlnum", false) }
    static var punctuation: Int { int("punctuation", 0) }  // 0「、。」1「，．」2「，。」
    static var candidateCount: Int { max(3, min(20, int("candidateCount", 9))) }
    static var fontSize: Int { max(11, min(28, int("fontSize", 15))) }

    /// 句読点を設定に合わせる
    static func punctuate(_ s: String) -> String {
        switch punctuation {
        case 1: return s.replacingOccurrences(of: "、", with: "，").replacingOccurrences(of: "。", with: "．")
        case 2: return s.replacingOccurrences(of: "、", with: "，")
        default: return s
        }
    }
}

// 設定画面の、保存しない状態（コマンドラインの swiftc では @State のマクロが使えないので、クラスに持たせる）
final class SettingsModel: ObservableObject {
    @Published var status = "確かめています…"
    @Published var confirmForget = false
    @Published var message = ""
}

// 設定画面（入力メニューの「設定…」から開く）
struct SettingsView: View {
    @ObservedObject var model: SettingsModel
    @AppStorage("liveConversion") var liveConversion = false
    @AppStorage("prediction") var prediction = true
    @AppStorage("useModel") var useModel = true
    @AppStorage("useContext") var useContext = true
    @AppStorage("learning") var learning = true
    @AppStorage("fullWidthSpace") var fullWidthSpace = false
    @AppStorage("fullWidthAlnum") var fullWidthAlnum = false
    @AppStorage("punctuation") var punctuation = 0
    @AppStorage("candidateCount") var candidateCount = 9
    @AppStorage("fontSize") var fontSize = 15

    var body: some View {
        Form {
            Section("変換") {
                Toggle("ライブ変換（スペースを押さなくても変換していく）", isOn: $liveConversion)
                Toggle("予測変換（打っている途中に候補を出す。Tab で選ぶ）", isOn: $prediction)
                Toggle("モデルで検算する（切ると速くなるが、精度は下がる）", isOn: $useModel)
                Toggle("前に確定した文を、文脈としてモデルに渡す", isOn: $useContext)
            }
            Section("入力") {
                Picker("句読点", selection: $punctuation) {
                    Text("、。").tag(0)
                    Text("，．").tag(1)
                    Text("，。").tag(2)
                }
                Toggle("スペースを全角で入れる（Shift＋スペースで半角）", isOn: $fullWidthSpace)
                Toggle("数字と英字（Shift）を全角で入れる", isOn: $fullWidthAlnum)
            }
            Section("候補の窓") {
                Stepper("一度に出す候補の数：\(candidateCount)", value: $candidateCount, in: 3...20)
                Stepper("文字の大きさ：\(fontSize)", value: $fontSize, in: 11...28)
            }
            Section("学習と辞書") {
                Toggle("学習を使う（確定した変換を、次から先に出す）", isOn: $learning)
                HStack {
                    Button("学習を消す…") { model.confirmForget = true }
                    Button("ユーザー辞書を編集…") { openUserDictionary() }
                    Button("置き場所を開く") { NSWorkspace.shared.open(Settings.storeFolder) }
                }
                if !model.message.isEmpty { Text(model.message).font(.caption).foregroundStyle(.secondary) }
            }
            Section("変換サーバー") {
                Text(model.status).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                Button("もう一度確かめる") { refresh() }
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 800)
        .onAppear { refresh() }
        .alert("学習を全部消しますか？", isPresented: $model.confirmForget) {
            Button("消す", role: .destructive) {
                Server.forget()
                model.message = "学習を消しました。"
            }
            Button("やめる", role: .cancel) {}
        } message: {
            Text("確定した変換の記録（history.tsv）を消します。元には戻せません。")
        }
    }

    private func refresh() {
        model.status = "確かめています…"
        DispatchQueue.global().async {
            let text: String
            if let s = Server.status() {
                let device = (s["device"] as? String) == "gpu" ? "GPU" : "CPU"
                text = """
                つながっています（127.0.0.1:\(s["port"] ?? "")）
                重み：\((s["run"] as? String).map { URL(fileURLWithPath: $0).lastPathComponent } ?? "") / \(s["ckpt"] ?? "")
                辞書：\((s["dict"] as? String).map { URL(fileURLWithPath: $0).lastPathComponent } ?? "")
                計算：\(device)（設定 \(s["device_setting"] ?? "")。学習中は CPU）
                """
            } else {
                text = "つながりません。変換サーバー（ime/server.py）が動いていません。\nひらがなとカタカナだけで変換します。"
            }
            DispatchQueue.main.async { model.status = text }
        }
    }

    private func openUserDictionary() {
        let file = Settings.storeFolder.appendingPathComponent("user_dict.tsv")
        if !FileManager.default.fileExists(atPath: file.path) {
            try? FileManager.default.createDirectory(at: Settings.storeFolder, withIntermediateDirectories: true)
            try? "# 読み（ひらがなかカタカナ）<TAB>変換。# で始まる行は飛ばす。UTF-8 で保存する\n".write(to: file, atomically: true, encoding: .utf8)
        }
        NSWorkspace.shared.open([file], withApplicationAt: URL(fileURLWithPath: "/System/Applications/TextEdit.app"),
                                configuration: NSWorkspace.OpenConfiguration())
        model.message = "UTF-8 のまま保存すれば、次の変換から使います。"
    }
}

final class SettingsWindow {
    static let shared = SettingsWindow()
    private var window: NSWindow?

    func show() {
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 560, height: 800),
                             styleMask: [.titled, .closable], backing: .buffered, defer: false)
            w.contentView = NSHostingView(rootView: SettingsView(model: SettingsModel()))
            w.title = "IXV かな漢字の設定"
            w.styleMask = [.titled, .closable]
            w.isReleasedWhenClosed = false
            w.center()
            window = w
        }
        NSLog("IXVKanji: show settings window")
        window?.level = .floating
        NSApp.activate()
        window?.makeKeyAndOrderFront(nil)
        window?.orderFrontRegardless()
    }
}
