import Cocoa
import InputMethodKit

// 入力メソッドの起動。接続名は Info.plist の InputMethodConnectionName。
let connectionName = Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String ?? "jp.co.elvez.inputmethod.IXVKanji_Connection"
let server = IMKServer(name: connectionName, bundleIdentifier: Bundle.main.bundleIdentifier)
_ = server
NSApplication.shared.run()
