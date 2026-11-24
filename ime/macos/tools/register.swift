// 入れ直した入力メソッドを、入力ソースとして登録し直す。Info.plist の変更（アイコンなど）を早く反映させるため。
import Carbon
import Foundation

let url = URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL
let status = TISRegisterInputSource(url)
print(status == noErr ? "registered" : "TISRegisterInputSource: \(status)")
