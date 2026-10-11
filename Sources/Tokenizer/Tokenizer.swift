// Qwen2.5 byte-level BPE tokenizer, bit-exact with HuggingFace `tokenizers` (Oniguruma split regex,
// unicode-normalization-alignments NFC). `Tokenizer.compile` turns tokenizer.json into a compact
// binary file; `init(path:)` mmaps it (no parsing, no dictionary building).
import Darwin

public enum TokenizerError: Error, CustomStringConvertible {
    case io(String)
    case format(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .io(let s): return "tokenizer I/O error: \(s)"
        case .format(let s): return "tokenizer format error: \(s)"
        case .unsupported(let s): return "unsupported tokenizer: \(s)"
        }
    }
}

// Compiled file header: 64 little-endian UInt32 words.
enum H {
    static let magic: UInt32 = 0x314B_5451  // "QTK1"
    static let version: UInt32 = 1
    static let words = 64
    static let size = words * 4
    // field indices
    static let fMagic = 0, fVersion = 1, fFileSize = 2, fNVocab = 3, fNTotal = 4, fNMerges = 5
    static let fMaxTokenLen = 6, fImStart = 7, fImEnd = 8, fEndOfText = 9, fMergeBits = 10
    static let fTokenBits = 11, fNSpecials = 12, fNUniBlocks = 13, fNDecomp = 14, fNDecompData = 15
    static let fNComp = 16, fSpecFirstByte = 17
    static let oTokOff = 20, oTokBytes = 21, oByteTok = 22, oByteProp = 23, oMerge = 24, oTokTab = 25
    static let oSpecials = 26, oSpecBytes = 27, oUni1 = 28, oUni2 = 29, oDecompKeys = 30
    static let oDecompVals = 31, oDecompData = 32, oCompKeys = 33, oCompVals = 34
}

public final class Tokenizer: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let mappedSize: Int
    let core: Core
    let nTotal: Int
    let specTab: UnsafePointer<UInt32>   // per special: id, offset into specBytes, length
    let specBytes: UnsafePointer<UInt8>
    let nSpecials: Int
    let specFirstByte: Int32             // shared first byte of all specials, or -1
    var sink: UInt8 = 0

    public let imStart: Int32
    public let imEnd: Int32
    public let endOfText: Int32

    /// Number of token ids (BPE vocab + added tokens).
    public var vocabSize: Int { nTotal }

    /// mmap a file produced by `Tokenizer.compile`.
    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw TokenizerError.io("cannot open \(path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw TokenizerError.io("cannot stat \(path)") }
        let size = Int(st.st_size)
        guard size >= H.size else { throw TokenizerError.format("\(path): file too small") }
        guard let m = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), m != MAP_FAILED else {
            throw TokenizerError.io("mmap failed for \(path)")
        }
        let hdr = UnsafeRawPointer(m).assumingMemoryBound(to: UInt32.self)
        guard hdr[H.fMagic] == H.magic, hdr[H.fVersion] == H.version, Int(hdr[H.fFileSize]) == size else {
            munmap(m, size)
            throw TokenizerError.format("\(path): not a compiled tokenizer (or wrong version); recompile it")
        }
        base = m
        mappedSize = size
        let raw = UnsafeRawPointer(m)
        func ptr<T>(_ field: Int, _: T.Type) -> UnsafePointer<T> {
            (raw + Int(hdr[field])).assumingMemoryBound(to: T.self)
        }
        let mergeBits = UInt64(hdr[H.fMergeBits]), tokenBits = UInt64(hdr[H.fTokenBits])
        core = Core(
            tokOff: ptr(H.oTokOff, UInt32.self), tokBytes: ptr(H.oTokBytes, UInt8.self),
            byteTok: ptr(H.oByteTok, Int32.self), byteProp: ptr(H.oByteProp, UInt16.self),
            mergeTab: ptr(H.oMerge, UInt64.self), mergeShift: 64 - mergeBits, mergeMask: (1 << Int(mergeBits)) - 1,
            tokTab: ptr(H.oTokTab, UInt64.self), tokShift: 64 - tokenBits, tokMask: (1 << Int(tokenBits)) - 1,
            maxTokenLen: Int(hdr[H.fMaxTokenLen]),
            uni1: ptr(H.oUni1, UInt16.self), uni2: ptr(H.oUni2, UInt16.self),
            decompKeys: ptr(H.oDecompKeys, UInt32.self), decompVals: ptr(H.oDecompVals, UInt32.self),
            decompData: ptr(H.oDecompData, UInt32.self), nDecomp: Int(hdr[H.fNDecomp]),
            compKeys: ptr(H.oCompKeys, UInt64.self), compVals: ptr(H.oCompVals, UInt32.self),
            nComp: Int(hdr[H.fNComp]))
        nTotal = Int(hdr[H.fNTotal])
        specTab = ptr(H.oSpecials, UInt32.self)
        specBytes = ptr(H.oSpecBytes, UInt8.self)
        nSpecials = Int(hdr[H.fNSpecials])
        specFirstByte = Int32(bitPattern: hdr[H.fSpecFirstByte])
        imStart = Int32(bitPattern: hdr[H.fImStart])
        imEnd = Int32(bitPattern: hdr[H.fImEnd])
        endOfText = Int32(bitPattern: hdr[H.fEndOfText])
    }

    deinit { munmap(base, mappedSize) }

    // MARK: Encoding

    /// Plain text: no added/special-token matching (a literal "<|im_end|>" is encoded as text).
    public func encode(_ text: String) -> [Int32] {
        withBytes(text) { encodeBytes($0, $1, specials: false) }
    }

    /// Added tokens (<|im_start|>, <|im_end|>, <|endoftext|>, <tool_call>, ...) are matched
    /// atomically, exactly like HF `Tokenizer.encode` (leftmost-longest, before normalization).
    public func encodeWithSpecials(_ text: String) -> [Int32] {
        withBytes(text) { encodeBytes($0, $1, specials: true) }
    }

    /// "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    /// tokenized like HF (apply_chat_template + encode): added tokens inside system/user are matched too.
    public func encodeChat(system: String, user: String) -> [Int32] {
        let s0 = tokenBytes(imStart), s1 = tokenBytes(imEnd)
        let lit0 = Array("system\n".utf8), lit1 = Array("\n".utf8), lit2 = Array("user\n".utf8)
        let lit3 = Array("assistant\n".utf8)
        return withBytes(system) { sp, sn in
            withBytes(user) { up, un in
                let total = s0.count * 3 + s1.count * 2 + lit0.count + lit1.count * 2 + lit2.count + lit3.count + sn + un
                return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: total) { buf in
                    var w = 0
                    let o = buf.baseAddress!
                    func put(_ p: UnsafeRawPointer?, _ n: Int) {
                        if n > 0, let p { (o + w).update(from: p.assumingMemoryBound(to: UInt8.self), count: n) }
                        w += n
                    }
                    func put(_ r: UnsafeRawBufferPointer) { put(r.baseAddress, r.count) }
                    func put(_ a: [UInt8]) { a.withUnsafeBytes { put($0) } }
                    put(s0); put(lit0); put(UnsafeRawPointer(sp), sn); put(s1); put(lit1)
                    put(s0); put(lit2); put(UnsafeRawPointer(up), un); put(s1); put(lit1)
                    put(s0); put(lit3)
                    return encodeBytes(UnsafePointer(o), w, specials: true)
                }
            }
        }
    }

    /// Encode UTF-8 bytes (must be valid UTF-8).
    public func encode(utf8 p: UnsafePointer<UInt8>, count n: Int, specials: Bool) -> [Int32] {
        encodeBytes(p, n, specials: specials)
    }

    @inline(__always)
    func withBytes<R>(_ s: String, _ body: (UnsafePointer<UInt8>, Int) -> R) -> R {
        if let r = s.utf8.withContiguousStorageIfAvailable({ b -> R in
            if let a = b.baseAddress { return body(a, b.count) }
            return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 1) { body($0.baseAddress!, 0) }
        }) {
            return r
        }
        var c = s
        return c.withUTF8 { b in
            if let a = b.baseAddress { return body(a, b.count) }
            return withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 1) { body($0.baseAddress!, 0) }
        }
    }

    func encodeBytes(_ p: UnsafePointer<UInt8>, _ n: Int, specials: Bool) -> [Int32] {
        if n == 0 { return [] }
        let core = self.core
        if !core.needsNFC(p, n) {
            // Fast path: text is already NFC; every piece yields at most one token per byte.
            return [Int32](unsafeUninitializedCapacity: n) { buf, count in
                var cnt = 0
                let out = buf.baseAddress!
                if specials {
                    splitSpecials(p, n) { a, b, id in
                        if id >= 0 {
                            out[cnt] = id
                            cnt += 1
                        } else {
                            core.encodeSegment(p + a, b - a, out, &cnt)
                        }
                    }
                } else {
                    core.encodeSegment(p, n, out, &cnt)
                }
                count = cnt
            }
        }
        return encodeSlow(p, n, specials: specials)
    }

    /// Text that needs NFC somewhere: normalize each non-special segment (HF matches added tokens on
    /// the raw text, then normalizes the remaining segments independently).
    func encodeSlow(_ p: UnsafePointer<UInt8>, _ n: Int, specials: Bool) -> [Int32] {
        let core = self.core
        var norm = [UInt8]()
        norm.reserveCapacity(n + n / 2 + 16)
        var segs = [(Int, Int, Int32)]()
        func addText(_ a: Int, _ b: Int) {
            let s = norm.count
            if core.needsNFC(p + a, b - a) {
                core.nfc(p + a, b - a, into: &norm)
            } else {
                norm.append(contentsOf: UnsafeBufferPointer(start: p + a, count: b - a))
            }
            segs.append((s, norm.count, -1))
        }
        if specials {
            splitSpecials(p, n) { a, b, id in
                if id >= 0 { segs.append((0, 0, id)) } else { addText(a, b) }
            }
        } else {
            addText(0, n)
        }
        return norm.withUnsafeBufferPointer { nb in
            [Int32](unsafeUninitializedCapacity: nb.count + segs.count + 1) { buf, count in
                var cnt = 0
                let out = buf.baseAddress!
                for (a, b, id) in segs {
                    if id >= 0 {
                        out[cnt] = id
                        cnt += 1
                    } else if b > a {
                        core.encodeSegment(nb.baseAddress! + a, b - a, out, &cnt)
                    }
                }
                count = cnt
            }
        }
    }

    /// Calls body(start, end, -1) for each text segment and body(0, 0, id) for each added token.
    @inline(__always)
    func splitSpecials(_ p: UnsafePointer<UInt8>, _ n: Int, _ body: (Int, Int, Int32) -> Void) {
        var segStart = 0
        var i = 0
        while i < n {
            var q: Int
            if specFirstByte >= 0 {
                guard let r = memchr(p + i, specFirstByte, n - i) else { break }
                q = UnsafeRawPointer(r) - UnsafeRawPointer(p)
            } else {
                q = i
            }
            if let (id, len) = matchSpecial(p, q, n) {
                if q > segStart { body(segStart, q, -1) }
                body(0, 0, id)
                i = q + len
                segStart = i
            } else {
                i = q + 1
            }
        }
        if segStart < n { body(segStart, n, -1) }
    }

    /// Longest added token starting at p[q], if any.
    @inline(__always)
    func matchSpecial(_ p: UnsafePointer<UInt8>, _ q: Int, _ n: Int) -> (Int32, Int)? {
        var bestLen = 0
        var bestId: Int32 = -1
        let rem = n - q
        for k in 0..<nSpecials {
            let len = Int(specTab[3 * k + 2])
            if len > rem || len <= bestLen { continue }
            let s = specBytes + Int(specTab[3 * k + 1])
            if s[0] == p[q] && bytesEqual(s, p + q, len) {
                bestLen = len
                bestId = Int32(bitPattern: specTab[3 * k])
            }
        }
        return bestId >= 0 ? (bestId, bestLen) : nil
    }

    // MARK: Decoding

    /// Raw bytes of a token (valid while the Tokenizer lives). Empty for ids without a token.
    @inline(__always)
    public func tokenBytes(_ id: Int32) -> UnsafeRawBufferPointer {
        guard id >= 0 && Int(id) < nTotal else { return UnsafeRawBufferPointer(start: nil, count: 0) }
        let s = Int(core.tokOff[Int(id)]), e = Int(core.tokOff[Int(id) + 1])
        return UnsafeRawBufferPointer(start: core.tokBytes + s, count: e - s)
    }

    /// Like HF decode(ids, skip_special_tokens=False): concatenated bytes, invalid UTF-8 replaced
    /// by U+FFFD; ids without a token are skipped.
    public func decode(_ ids: [Int32]) -> String {
        let off = core.tokOff
        var total = 0
        for id in ids where id >= 0 && Int(id) < nTotal {
            total += Int(off[Int(id) + 1] &- off[Int(id)])
        }
        if total == 0 { return "" }
        let bytes = core.tokBytes
        return String(unsafeUninitializedCapacity: total) { buf in
            var w = 0
            let o = buf.baseAddress!
            for id in ids where id >= 0 && Int(id) < nTotal {
                let s = Int(off[Int(id)]), len = Int(off[Int(id) + 1]) - s
                (o + w).update(from: bytes + s, count: len)
                w += len
            }
            return w
        }
    }

    // MARK: Diagnostics

    /// The NFC normalization HF applies before pre-tokenization.
    public func normalize(_ text: String) -> String {
        withBytes(text) { p, n in
            if !core.needsNFC(p, n) { return text }
            var out = [UInt8]()
            core.nfc(p, n, into: &out)
            return String(decoding: out, as: UTF8.self)
        }
    }

    /// Pre-tokenized pieces (after NFC) of plain text, as raw byte strings.
    public func preTokenize(_ text: String) -> [[UInt8]] {
        withBytes(normalize(text)) { p, n in
            var pieces = [[UInt8]]()
            var i = 0
            while i < n {
                let j = core.nextPieceEnd(p, i, n)
                pieces.append(Array(UnsafeBufferPointer(start: p + i, count: j - i)))
                i = j
            }
            return pieces
        }
    }

    /// Touch every page of the mapping (moves first-use page faults off the critical path).
    public func prefetch() {
        var acc: UInt8 = 0
        let p = UnsafeRawPointer(base).assumingMemoryBound(to: UInt8.self)
        var i = 0
        while i < mappedSize {
            acc &+= p[i]
            i += 4096
        }
        sink = acc
    }
}
