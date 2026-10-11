import Foundation

// q4a: w = s * (n - zq/4), n in 0...15, s = unsigned E5M3 (fp16 bits = code << 7), zq in 0...63.
// One group per (row, lane slice): 28 weights for K = 896, 38 for K = 4864 (split into 4 quarters).
// Layout per 16-row block (and quarter): NN = COLS/2 nibble chunks, then one scale chunk, then the
// zero points: a "chunk" is 512 contiguous bytes, lane l's 16 bytes at +16 l, byte r = row r.
//   nibble chunk c: byte r = n[r][2c] | n[r][2c+1] << 4   (columns of lane l's slice)
//   scale chunk:    byte r = E5M3 code of (row r, lane l)
//   zlo: lane l's UInt64 at (NN+1)*512 + 8 l, bits 4r..4r+3 = zq[r] & 15
//   zhi: lane l's UInt32 at (NN+1)*512 + 256 + 4 l, bits 2r..2r+1 = zq[r] >> 4

enum Q4A {
    static func e5m3Value(_ c: UInt8) -> Float { Float(Float16(bitPattern: UInt16(c) << 7)) }

    static func e5m3Nearest(_ s: Float) -> UInt8 {
        if !(s > 0) { return 0 }
        let b = UInt32(Float16(s).bitPattern)
        return UInt8(min(max((b + 0x40) >> 7, 1), 0xF7))
    }

    /// Searches scale (10 values around range/15) and zero point (5 offsets); returns (code, zq) and codes.
    @inline(__always)
    static func quantGroup(_ w: UnsafePointer<Float>, _ n: Int, _ out: UnsafeMutablePointer<UInt8>,
                           _ tmp: UnsafeMutablePointer<UInt8>) -> (UInt8, UInt8) {
        var mn = w[0], mx = w[0]
        for i in 1..<n { mn = min(mn, w[i]); mx = max(mx, w[i]) }
        let rng = max(mx - mn, 1e-20)
        var bestErr = Float.infinity
        var best: (UInt8, UInt8) = (0, 0)
        for fi in 0..<10 {
            let f = 0.75 + Float(fi) * (0.35 / 9)
            let code = e5m3Nearest(rng * f / 15)
            let s = e5m3Value(code)
            if s <= 0 { continue }
            let inv = 1 / s
            let z0 = 7.5 - (mn + mx) / (2 * s)
            for dzi in -2...2 {
                let zq = UInt8(min(63, max(0, (4 * (z0 + Float(dzi) * 0.25)).rounded())))
                let z = Float(zq) * 0.25
                var err: Float = 0
                for i in 0..<n {
                    let q = min(15, max(0, (w[i] * inv + z).rounded()))
                    tmp[i] = UInt8(q)
                    let d = s * (q - z) - w[i]
                    err += d * d
                }
                if err < bestErr {
                    bestErr = err
                    best = (code, zq)
                    out.update(from: tmp, count: n)
                }
            }
        }
        return best
    }

    /// Packs a row-major [rows][K] matrix (rows multiple of 16).
    static func pack(_ m: [Float], rows: Int, K: Int) -> [UInt8] {
        let (COLS, S) = K == 896 ? (28, 1) : (38, 4)
        precondition(K == 896 || K == 4864)
        let NN = COLS / 2, BLKB = (NN + 1) * 512 + 384, nb = rows / 16
        var out = [UInt8](repeating: 0, count: nb * S * BLKB)
        m.withUnsafeBufferPointer { wp in
            out.withUnsafeMutableBufferPointer { op in
                DispatchQueue.concurrentPerform(iterations: nb) { b in
                    var nib = [UInt8](repeating: 0, count: COLS)
                    var tmp = [UInt8](repeating: 0, count: COLS)
                    for s in 0..<S {
                        let base = (b * S + s) * BLKB
                        for l in 0..<32 {
                            var zlo: UInt64 = 0, zhi: UInt32 = 0
                            for r in 0..<16 {
                                let row = b * 16 + r, col0 = s * (K / S) + l * COLS
                                let (code, zq) = nib.withUnsafeMutableBufferPointer { nb in
                                    tmp.withUnsafeMutableBufferPointer { tb in
                                        quantGroup(wp.baseAddress! + row * K + col0, COLS, nb.baseAddress!, tb.baseAddress!)
                                    }
                                }
                                for c in 0..<NN { op[base + (c * 32 + l) * 16 + r] = nib[2 * c] | (nib[2 * c + 1] << 4) }
                                op[base + (NN * 32 + l) * 16 + r] = code
                                zlo |= UInt64(zq & 0xF) << UInt64(4 * r)
                                zhi |= UInt32(zq >> 4) << UInt32(2 * r)
                            }
                            let zo = base + (NN + 1) * 512 + l * 8
                            for i in 0..<8 { op[zo + i] = UInt8((zlo >> UInt64(8 * i)) & 0xFF) }
                            let ho = base + (NN + 1) * 512 + 256 + l * 4
                            for i in 0..<4 { op[ho + i] = UInt8((zhi >> UInt32(8 * i)) & 0xFF) }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Stored QKV row i = original (q||k||v) row qkvOrder()[i]: blocks 0..63 pair rotary dims
    /// (rows 0-7 = dims 8m..8m+7, rows 8-15 = dims 32+8m..) of q/k heads; blocks 64..71 = v rows.
    static func qkvOrder() -> [Int] {
        var o: [Int] = []
        for b in 0..<64 {
            let head = b / 4, m = b % 4
            for i in 0..<8 { o.append(64 * head + 8 * m + i) }
            for i in 0..<8 { o.append(64 * head + 32 + 8 * m + i) }
        }
        for b in 64..<72 { for i in 0..<16 { o.append(1024 + 16 * (b - 64) + i) } }
        return o
    }

    /// Stored gate||up row: block b = gate rows 8b..8b+7 then up rows 8b..8b+7.
    static func guOrder(ff: Int) -> [Int] {
        var o: [Int] = []
        for b in 0..<(ff / 8) {
            for i in 0..<8 { o.append(8 * b + i) }
            for i in 0..<8 { o.append(ff + 8 * b + i) }
        }
        return o
    }
}
