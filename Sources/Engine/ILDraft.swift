import Foundation
import Metal

// The learned speculative drafter (reference: tools/draft/ildraft.py, table format ILDRAFT1).
// Static tables live in one GPU buffer (the whole file) plus a dense token-flag table; the
// per-request context (TF-IDF neighbours' outputs, copy candidates, prompt-word associations) is
// prepared on the CPU before generation and read by k_ildraft, which drafts on the GPU.

public final class ILDraft {
    public let N, M, KNB, MAXM, F, H, flags, nDocs: Int
    let data: Data
    let sections: [String: (count: Int, offset: Int)]
    public let table: MTLBuffer     // the file as is
    public let tflg: MTLBuffer      // u8[VOCAB]: '`' count (bits 0-2), newline (bit 3)
    public let meta: MTLBuffer      // section offsets for the kernel (DraftMeta)
    public let req: MTLBuffer       // per-request context (DraftReq)
    var docScore: [Double]
    var touched: [Int32] = []

    static let tail: [Int32] = [151645, 198, 151644, 77091, 198]

    // DraftReq layout (Int32 slots), mirrored in Shaders.swift.
    static let maxP = 256, maxCopy = 256, maxNB = 16, nbSeqMax = 80, maxAs = 1024
    static let oNP = 0, oNCopy = 1, oNNB = 2, oNAs = 3
    static let oP = 4
    static let oCopyTok = oP + maxP
    static let oCopyInP = oCopyTok + maxCopy
    static let oNBScore = oCopyInP + maxCopy
    static let oNBOff = oNBScore + maxNB
    static let oNBTok = oNBOff + maxNB + 1
    static let oAsTok = oNBTok + maxNB * nbSeqMax
    static let oAsMax = oAsTok + maxAs
    static let oAsCnt = oAsMax + maxAs
    static let reqInts = oAsCnt + maxAs

    public init(path: String, device: MTLDevice) throws {
        guard let d = FileManager.default.contents(atPath: path), d.count >= 64 else { throw EngineError("cannot read \(path)") }
        data = d
        let h = d.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self).prefix(16)) }
        guard d.prefix(8) == "ILDRAFT1".data(using: .ascii)!, h[2] == 1 else { throw EngineError("\(path): not an ILDRAFT1 table") }
        let nsec = Int(h[3])
        N = Int(h[4]); M = Int(h[5]); KNB = Int(h[6]); MAXM = Int(h[7]); F = Int(h[8]); H = Int(h[9])
        nDocs = Int(h[11]); flags = Int(h[12])
        guard F == 53, N <= 6, M <= 7, KNB <= ILDraft.maxNB, MAXM <= 12, H <= 32 else { throw EngineError("\(path): unsupported drafter config") }
        var secs: [String: (Int, Int)] = [:]
        d.withUnsafeBytes { raw in
            for i in 0..<nsec {
                let b = 64 + 16 * i
                let tag = String(decoding: raw[b..<(b + 4)], as: UTF8.self)
                let cnt = Int(raw.load(fromByteOffset: b + 4, as: UInt32.self))
                let off = Int(raw.loadUnaligned(fromByteOffset: b + 8, as: UInt64.self))
                secs[tag] = (cnt, off)
            }
        }
        sections = secs.mapValues { (count: $0.0, offset: $0.1) }
        for t in ["NGKY", "NGVL", "NGNX", "WDKY", "WDVL", "POST", "DNRM", "SQOF", "SQTK", "CPTK", "CPC1", "CPC2", "TFLS", "MLP0"] {
            guard sections[t] != nil else { throw EngineError("\(path): missing section \(t)") }
        }
        table = d.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
        var tf = [UInt8](repeating: 0, count: Dims.vocab)
        let (tfc, tfo) = (sections["TFLS"]!.count, sections["TFLS"]!.offset)
        d.withUnsafeBytes { raw in
            for i in 0..<tfc {
                let tok = Int(raw.load(fromByteOffset: tfo + 8 * i, as: UInt32.self))
                if tok < Dims.vocab { tf[tok] = UInt8(raw.load(fromByteOffset: tfo + 8 * i + 4, as: UInt32.self)) }
            }
        }
        tflg = device.makeBuffer(bytes: tf, length: tf.count, options: .storageModeShared)!
        // DraftMeta: byte offsets / counts of the sections the kernel reads, then the config.
        var m = [UInt32](repeating: 0, count: 32)
        let secMap = sections
        let put = { (i: Int, t: String) in m[2 * i] = UInt32(secMap[t]!.offset); m[2 * i + 1] = UInt32(secMap[t]!.count) }
        put(0, "NGKY"); put(1, "NGVL"); put(2, "NGNX"); put(3, "CPTK"); put(4, "CPC1"); put(5, "CPC2"); put(6, "MLP0")
        m[16] = UInt32(N); m[17] = UInt32(M); m[18] = UInt32(MAXM); m[19] = UInt32(H)
        meta = device.makeBuffer(bytes: m, length: m.count * 4, options: .storageModeShared)!
        req = device.makeBuffer(length: ILDraft.reqInts * 4, options: .storageModeShared)!
        docScore = [Double](repeating: 0, count: nDocs)
    }

    // MARK: table access (CPU)

    @inline(__always) private func u32(_ off: Int) -> UInt32 { data.withUnsafeBytes { $0.load(fromByteOffset: off, as: UInt32.self) } }
    @inline(__always) private func f32(_ off: Int) -> Float { data.withUnsafeBytes { $0.load(fromByteOffset: off, as: Float.self) } }
    @inline(__always) private func u64(_ off: Int) -> UInt64 { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt64.self) } }

    /// Index of `key` in a sorted u64 key section, or nil.
    private func find(_ tag: String, _ key: UInt64) -> Int? {
        let (cnt, off) = sections[tag]!
        var lo = 0, hi = cnt
        while lo < hi {
            let mid = (lo + hi) / 2
            let k = u64(off + 8 * mid)
            if k < key { lo = mid + 1 } else { hi = mid }
        }
        return lo < cnt && u64(off + 8 * lo) == key ? lo : nil
    }

    static func fnv1a64(_ b: [UInt8]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for x in b { h ^= UInt64(x); h = h &* 0x100_0000_01b3 }
        return h
    }

    /// Distinct runs of [a-z0-9] of the ASCII-lowercased UTF-8 text, in first-appearance order.
    static func promptWords(_ text: String) -> [[UInt8]] {
        var out: [[UInt8]] = []
        var seen = Set<[UInt8]>()
        var cur: [UInt8] = []
        for var c in Array(text.utf8) + [0] {
            if c >= 65 && c <= 90 { c += 32 }
            if (c >= 97 && c <= 122) || (c >= 48 && c <= 57) { cur.append(c); continue }
            if !cur.isEmpty { if seen.insert(cur).inserted { out.append(cur) }; cur.removeAll(keepingCapacity: true) }
        }
        return out
    }

    /// Runs of [A-Za-z0-9_.\-/~:@] (ASCII) of the text, in order (space-variant candidates).
    static func spaceVarWords(_ text: String) -> [String] {
        var out: [String] = []
        var cur: [UInt8] = []
        let ok = { (c: UInt8) -> Bool in
            (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || (c >= 48 && c <= 57) || c == 95 || c == 46 || c == 45 || c == 47 || c == 126 || c == 58 || c == 64
        }
        for c in Array(text.utf8) + [0] {
            if ok(c) { cur.append(c); continue }
            if !cur.isEmpty { out.append(String(decoding: cur, as: UTF8.self)); cur.removeAll(keepingCapacity: true) }
        }
        return out
    }

    // MARK: per-request context

    /// Prepares the request context: `prompt` = the user prompt's token ids inside the chat
    /// template, `text` = the user prompt, `encode` = the tokenizer (no special tokens).
    public func prepare(prompt: [Int32], text: String, encode: (String) -> [Int32]) {
        let r = req.contents().assumingMemoryBound(to: Int32.self)
        let rf = req.contents().assumingMemoryBound(to: Float.self)
        let P = Array(prompt.prefix(ILDraft.maxP))
        r[ILDraft.oNP] = Int32(P.count)
        for (i, t) in P.enumerated() { r[ILDraft.oP + i] = t }
        let pset = Set(P)
        // neighbours: TF-IDF cosine over prompt words, top KNB by (score desc, doc asc)
        let words = ILDraft.promptWords(text)
        let (wvOff, poOff) = (sections["WDVL"]!.offset, sections["POST"]!.offset)
        var q2: Double = 0      // float64 like the reference, so neighbour order matches
        for i in touched { docScore[Int(i)] = 0 }
        touched.removeAll(keepingCapacity: true)
        for w in words {
            guard let k = find("WDKY", ILDraft.fnv1a64(w)) else { continue }
            let idf = Double(f32(wvOff + 12 * k)), first = Int(u32(wvOff + 12 * k + 4)), n = Int(u32(wvOff + 12 * k + 8))
            let w2 = idf * idf
            q2 += w2
            for j in 0..<n {
                let doc = Int(u32(poOff + 4 * (first + j)))
                if docScore[doc] == 0 { touched.append(Int32(doc)) }
                docScore[doc] += w2
            }
        }
        let qn = q2 > 0 ? q2.squareRoot() : 1
        let normOff = sections["DNRM"]!.offset
        var ranked: [(Double, Int32)] = touched.map { d in (docScore[Int(d)] / (qn * Double(f32(normOff + 4 * Int(d)))), d) }
        ranked.sort { $0.0 != $1.0 ? $0.0 > $1.0 : $0.1 < $1.1 }
        let (sqOff, sqTok) = (sections["SQOF"]!.offset, sections["SQTK"]!.offset)
        let nb = min(KNB, ranked.count)
        r[ILDraft.oNNB] = Int32(nb)
        var pos = 0
        for i in 0..<nb {
            let d = Int(ranked[i].1)
            rf[ILDraft.oNBScore + i] = Float(ranked[i].0)
            r[ILDraft.oNBOff + i] = Int32(pos)
            var seq = ILDraft.tail
            let a = Int(u32(sqOff + 4 * d)), b = Int(u32(sqOff + 4 * (d + 1)))
            for j in a..<b { seq.append(Int32(bitPattern: u32(sqTok + 4 * j))) }
            for t in seq.prefix(ILDraft.nbSeqMax) { r[ILDraft.oNBTok + pos] = t; pos += 1 }
        }
        r[ILDraft.oNBOff + nb] = Int32(pos)
        // copy candidates: prompt tokens and the first token of " word" for space-variant words
        var spacevar = Set<Int32>()
        for w in ILDraft.spaceVarWords(text) {
            if let t = encode(" " + w).first, !pset.contains(t) { spacevar.insert(t) }
        }
        let copy = Array(pset.union(spacevar)).sorted().prefix(ILDraft.maxCopy)
        r[ILDraft.oNCopy] = Int32(copy.count)
        for (i, t) in copy.enumerated() { r[ILDraft.oCopyTok + i] = t; r[ILDraft.oCopyInP + i] = pset.contains(t) ? 1 : 0 }
        // prompt-word -> output-token associations: (max loglift, number of words)
        var asMap: [Int32: (Float, Int32)] = [:]
        if flags & 1 != 0, let (awvOff, astOff) = sections["ASWV"].flatMap({ v in sections["ASTK"].map { (v.offset, $0.offset) } }) {
            for w in words {
                guard let k = find("ASWK", ILDraft.fnv1a64(w)) else { continue }
                let first = Int(u32(awvOff + 8 * k)), n = Int(u32(awvOff + 8 * k + 4))
                for j in 0..<n {
                    let tok = Int32(bitPattern: u32(astOff + 8 * (first + j))), ll = f32(astOff + 8 * (first + j) + 4)
                    if let e = asMap[tok] { asMap[tok] = (max(e.0, ll), e.1 + 1) } else { asMap[tok] = (ll, 1) }
                }
            }
        }
        let asl = asMap.keys.sorted().prefix(ILDraft.maxAs)
        r[ILDraft.oNAs] = Int32(asl.count)
        for (i, t) in asl.enumerated() {
            r[ILDraft.oAsTok + i] = t
            rf[ILDraft.oAsMax + i] = asMap[t]!.0
            r[ILDraft.oAsCnt + i] = asMap[t]!.1
        }
    }
}
