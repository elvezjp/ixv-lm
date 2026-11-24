import Cocoa
import InputMethodKit

// 候補の窓。カーソルの下に、番号つきで縦に並べる。選んでいる候補に色を付ける。
final class CandidatesPanel {
    static let shared = CandidatesPanel()

    private let panel: NSPanel
    private let stack = NSStackView()

    private init() {
        panel = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.level = .popUpMenu
        panel.hasShadow = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        let box = NSVisualEffectView()
        box.material = .menu
        box.state = .active
        box.wantsLayer = true
        box.layer?.cornerRadius = 6
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 2
        stack.edgeInsets = NSEdgeInsets(top: 4, left: 4, bottom: 4, right: 4)
        stack.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: box.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: box.trailingAnchor),
            stack.topAnchor.constraint(equalTo: box.topAnchor),
            stack.bottomAnchor.constraint(equalTo: box.bottomAnchor),
        ])
        panel.contentView = box
    }

    func show(_ candidates: [String], selected: Int, client: IMKTextInput, numbered: Bool = true) {
        stack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        // 設定の数ずつ、選んでいる候補を含む頁を出す。番号は頁の中で 1 から
        let size = Settings.candidateCount
        let start = max(0, selected) / size * size
        for (i, text) in candidates.enumerated() where i >= start && i < start + size {
            let n = i - start
            let label = NSTextField(labelWithString: !numbered ? text : n < 9 ? "\(n + 1)  \(text)" : "    \(text)")
            label.font = .systemFont(ofSize: CGFloat(Settings.fontSize))
            label.wantsLayer = true
            label.drawsBackground = true
            label.backgroundColor = i == selected ? .selectedContentBackgroundColor : .clear
            label.textColor = i == selected ? .alternateSelectedControlTextColor : .labelColor
            stack.addArrangedSubview(label)
        }
        if candidates.count > size {
            let page = NSTextField(labelWithString: "\(start / size + 1) / \((candidates.count + size - 1) / size)")
            page.font = .systemFont(ofSize: CGFloat(Settings.fontSize) - 3)
            page.textColor = .secondaryLabelColor
            stack.addArrangedSubview(page)
        }
        panel.setContentSize(stack.fittingSize)
        var rect = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &rect)
        panel.setFrameTopLeftPoint(NSPoint(x: rect.minX, y: rect.minY - 4))
        panel.orderFrontRegardless()
    }

    func hide() {
        panel.orderOut(nil)
    }
}
