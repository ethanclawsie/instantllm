import Foundation

// n-gram draft table for speculative decoding (layout mirrors NGEntry in Shaders.swift): built from
// the model's own greedy outputs; for every context of the last 1...6 generated tokens (padded with
// -1 before the output starts) it stores the most frequent next token, its count and the total.

public enum NGramTable {
    public static let slots = 1 << 20
    public static let maxN = 6

    @inline(__always)
    static func key(_ ctx: ArraySlice<Int32>) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325 ^ UInt64(ctx.count)
        for t in ctx {
            h ^= UInt64(UInt32(bitPattern: t)) &+ 0x9e37_79b9_7f4a_7c15
            h = h &* 0x0000_0100_0000_01b3
            h ^= h >> 29
        }
        return h | 1
    }

    /// Builds the table from sequences of generated token ids (each ending with its stop token).
    public static func build(_ seqs: [[Int32]]) -> Data {
        var counts: [UInt64: [Int32: Int]] = [:]
        for seq in seqs {
            let s = [Int32](repeating: -1, count: maxN) + seq
            for i in maxN..<s.count {
                for n in 1...maxN {
                    counts[key(s[(i - n)..<i]), default: [:]][s[i], default: 0] += 1
                }
            }
        }
        var table = Data(count: slots * 16)
        var used = 0
        table.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let p = raw.baseAddress!
            for (k, nexts) in counts {
                let total = nexts.values.reduce(0, +)
                let (tok, c) = nexts.max { a, b in a.value != b.value ? a.value < b.value : a.key > b.key }!
                var slot = Int((k >> 12) & UInt64(slots - 1))
                var placed = false
                for _ in 0..<16 {
                    if p.load(fromByteOffset: slot * 16, as: UInt64.self) == 0 {
                        p.storeBytes(of: k, toByteOffset: slot * 16, as: UInt64.self)
                        p.storeBytes(of: tok, toByteOffset: slot * 16 + 8, as: Int32.self)
                        p.storeBytes(of: UInt16(min(c, 65535)), toByteOffset: slot * 16 + 12, as: UInt16.self)
                        p.storeBytes(of: UInt16(min(total, 65535)), toByteOffset: slot * 16 + 14, as: UInt16.self)
                        placed = true
                        used += 1
                        break
                    }
                    slot = (slot + 1) & (slots - 1)
                }
                _ = placed
            }
        }
        return table
    }
}
