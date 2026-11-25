// メニューバーの入力ソースのアイコンを icon.tiff に書く。IXV の立方体のマーク（ixv-mark.svg）から「あ」を抜いたもの（案 a、2026-11-24 決定）。
// 透明の地に黒で描く。Info.plist の TISIconIsTemplate で、macOS が地の明るさに合わせて色を変える（濃さは不透明度で出す）。
// 使い方: swift make_icon.swift 出力.tiff [案 a|b|c] [見本.png か -] [仮名（英字のモードは A）]
import AppKit

let out = CommandLine.arguments[1]
let variant = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "a"
let kana = CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : "あ"
// 立方体の幅（16pt のうち）。元のマークの比率なら 13.9
let cubeWidth = CommandLine.arguments.count > 5 ? CGFloat(Double(CommandLine.arguments[5])!) : 13.9
let preview = CommandLine.arguments.count > 3 && CommandLine.arguments[3] != "-" ? CommandLine.arguments[3] : nil  // 「-」は見本なし

// ixv-mark.svg の 3 面（viewBox 174×200、y は下向き）
let top: [CGPoint] = [.init(x: 0.8, y: 50.5), .init(x: 87, y: 100.3), .init(x: 173.3, y: 50.5), .init(x: 87, y: 0.7)]
let left: [CGPoint] = [.init(x: 87, y: 100.3), .init(x: 0.8, y: 50.5), .init(x: 0.8, y: 150), .init(x: 87, y: 199.8)]
let right: [CGPoint] = [.init(x: 87, y: 100.3), .init(x: 173.3, y: 50.5), .init(x: 173.3, y: 150), .init(x: 87, y: 199.8)]

func draw(in ctx: CGContext, side: CGFloat) {
    // 16×16 いっぱいに置く（macOS の「あ」のアイコンも 16×16 いっぱいに描いている）
    let h: CGFloat = side, w = side * cubeWidth / 16
    let ox = (side - w) / 2, oy = (side - h) / 2
    func path(_ pts: [CGPoint]) -> CGPath {
        let p = CGMutablePath()
        p.addLines(between: pts.map { CGPoint(x: ox + $0.x / 174 * w, y: oy + h - $0.y / 200 * h) })
        p.closeSubpath()
        return p
    }
    switch variant {
    case "a":  // 立方体の 3 面を濃さで塗り分け、真ん中から仮名（あ・い・か）を抜く
        for (face, alpha) in [(top, 0.45), (left, 0.75), (right, 1.0)] {
            ctx.setFillColor(NSColor.black.withAlphaComponent(alpha).cgColor)
            ctx.addPath(path(face)); ctx.fillPath()
        }
        ctx.setBlendMode(.clear)
        let font = NSFont.boldSystemFont(ofSize: side * 12 / 16)
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: kana, attributes: [.font: font]))
        let b = CTLineGetBoundsWithOptions(line, .useGlyphPathBounds)
        ctx.textPosition = CGPoint(x: (side - b.width) / 2 - b.minX, y: (side - b.height) / 2 - b.minY - side / 32)
        CTLineDraw(line, ctx)
    case "c":  // 立方体の面を、すき間で分ける
        ctx.setFillColor(NSColor.black.cgColor)
        for face in [top, left, right] { ctx.addPath(path(face)) }
        ctx.fillPath()
        ctx.setBlendMode(.clear)
        ctx.setLineWidth(side / 16 * 1.2)
        let pts = [left[1], left[0], top[2]].map { CGPoint(x: ox + $0.x / 174 * w, y: oy + h - $0.y / 200 * h) }
        ctx.addLines(between: pts); ctx.strokePath()
        ctx.move(to: CGPoint(x: ox + w / 2, y: oy + h - 100.3 / 200 * h))
        ctx.addLine(to: CGPoint(x: ox + w / 2, y: oy)); ctx.strokePath()
    default:  // b：立方体の形を塗り、真ん中の「漢」を抜く
        ctx.setFillColor(NSColor.black.cgColor)
        for face in [top, left, right] { ctx.addPath(path(face)) }
        ctx.fillPath()
        ctx.setBlendMode(.clear)
        let font = NSFont.boldSystemFont(ofSize: side * 9.5 / 16)
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: "漢", attributes: [.font: font]))
        let b = CTLineGetBoundsWithOptions(line, .useGlyphPathBounds)
        ctx.textPosition = CGPoint(x: (side - b.width) / 2 - b.minX, y: (side - b.height) / 2 - b.minY)
        CTLineDraw(line, ctx)
    }
    ctx.setBlendMode(.normal)
}

func rep(_ px: Int) -> NSBitmapImageRep {
    let r = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
                             samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                             bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    let g = NSGraphicsContext(bitmapImageRep: r)!
    NSGraphicsContext.current = g
    g.cgContext.scaleBy(x: CGFloat(px) / 16, y: CGFloat(px) / 16)
    draw(in: g.cgContext, side: 16)
    g.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    r.size = NSSize(width: 16, height: 16)  // 描いた後で、点の大きさを 16pt にする（32px は Retina 用）
    return r
}

let reps = [rep(16), rep(32)]
try! NSBitmapImageRep.tiffRepresentationOfImageReps(in: reps, using: .none, factor: 0)!.write(to: URL(fileURLWithPath: out))

// 見本：明るい地に黒、暗い地に白で、32px（Retina の 16pt）を 4 倍に拡大して並べる
if let preview {
    let icon = reps[1].cgImage!
    let s = 128, pad = 24
    let img = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: s * 2 + pad * 3, pixelsHigh: s + pad * 2, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    let ctx = NSGraphicsContext(bitmapImageRep: img)!.cgContext
    ctx.interpolationQuality = .none
    for (i, (bg, fg)) in [(NSColor(white: 0.93, alpha: 1), NSColor.black), (NSColor(white: 0.17, alpha: 1), NSColor.white)].enumerated() {
        let x = pad + i * (s + pad)
        ctx.setFillColor(bg.cgColor); ctx.fill(CGRect(x: x - pad / 2, y: 0, width: s + pad, height: s + pad * 2))
        let rect = CGRect(x: x, y: pad, width: s, height: s)
        ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        ctx.draw(icon, in: rect)
        ctx.setBlendMode(.sourceIn); ctx.setFillColor(fg.cgColor); ctx.fill(rect)
        ctx.endTransparencyLayer(); ctx.setBlendMode(.normal)
    }
    NSGraphicsContext.restoreGraphicsState()
    try! img.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: preview))
}
