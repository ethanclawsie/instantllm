import Foundation

// Converts the Hugging Face bf16 safetensors checkpoint into an .ill file: reorders rows into the
// blocks the kernels expect and quantizes.

final class SafeTensors {
    struct Entry { let dtype: String; let shape: [Int]; let start: Int; let end: Int }
    let base: UnsafeRawPointer
    let length: Int
    let dataStart: Int
    let entries: [String: Entry]

    init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError("cannot open \(path)") }
        defer { close(fd) }
        var sb = stat()
        fstat(fd, &sb)
        length = Int(sb.st_size)
        guard let p = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else { throw EngineError("mmap \(path)") }
        base = UnsafeRawPointer(p)
        let n = Int(base.loadUnaligned(as: UInt64.self))
        dataStart = 8 + n
        let json = try JSONSerialization.jsonObject(with: Data(bytes: base.advanced(by: 8), count: n)) as! [String: Any]
        var e: [String: Entry] = [:]
        for (k, v) in json where k != "__metadata__" {
            let d = v as! [String: Any]
            let off = d["data_offsets"] as! [Int]
            e[k] = Entry(dtype: d["dtype"] as! String, shape: d["shape"] as! [Int], start: off[0], end: off[1])
        }
        entries = e
    }

    func has(_ name: String) -> Bool { entries[name] != nil }

    func raw(_ name: String) throws -> (shape: [Int], dtype: String, ptr: UnsafeRawPointer, count: Int) {
        guard let e = entries[name] else { throw EngineError("missing tensor \(name)") }
        return (e.shape, e.dtype, base.advanced(by: dataStart + e.start), e.end - e.start)
    }

    /// Any float tensor (BF16, F16, F32) as [Float].
    func anyFloats(_ name: String) throws -> [Float] {
        let r = try raw(name)
        switch r.dtype {
        case "F32": return Array(UnsafeBufferPointer(start: r.ptr.assumingMemoryBound(to: Float.self), count: r.count / 4))
        case "F16": return UnsafeBufferPointer(start: r.ptr.assumingMemoryBound(to: Float16.self), count: r.count / 2).map { Float($0) }
        default: return try floats(name).data
        }
    }

    func floats(_ name: String) throws -> (shape: [Int], data: [Float]) {
        guard let e = entries[name] else { throw EngineError("missing tensor \(name)") }
        guard e.dtype == "BF16" else { throw EngineError("\(name): expected BF16, got \(e.dtype)") }
        let count = (e.end - e.start) / 2
        let src = base.advanced(by: dataStart + e.start)
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                let b = src.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)
                dst[i] = Float(bitPattern: UInt32(b) << 16)
            }
        }
        return (e.shape, out)
    }
}

/// A 4-bit matrix: packed codes [rows][cols/2] and interleaved (scale, bias) f16 pairs [rows][cols/64][2].
struct Q4Mat {
    var packed: [UInt8]
    var sb: [Float16]
    let rows: Int
    let cols: Int

    func reordered(_ order: [Int]) -> Q4Mat {
        let pr = cols / 2, sr = cols / 64 * 2
        var p = [UInt8](repeating: 0, count: order.count * pr)
        var b = [Float16](repeating: 0, count: order.count * sr)
        for (i, r) in order.enumerated() {
            for j in 0..<pr { p[i * pr + j] = packed[r * pr + j] }
            for j in 0..<sr { b[i * sr + j] = sb[r * sr + j] }
        }
        return Q4Mat(packed: p, sb: b, rows: order.count, cols: cols)
    }

    static func + (a: Q4Mat, b: Q4Mat) -> Q4Mat {
        Q4Mat(packed: a.packed + b.packed, sb: a.sb + b.sb, rows: a.rows + b.rows, cols: a.cols)
    }
}

public enum Converter {
    public static func convert(hfDir: String, output: String, format: WeightFormat, quantFile: String? = nil,
                               log: (String) -> Void = { print($0) }) throws {
        let hf = try SafeTensors(path: hfDir + "/model.safetensors")
        let qf = try quantFile.map { try SafeTensors(path: $0) }
        if qf != nil && format != .q4 && format != .q4a { throw EngineError("--quant files hold q4 codes; use --format q4") }
        // Unquantized tensors (norms, biases): the quant file may override them (folded transforms).
        func vec(_ name: String) throws -> [Float] {
            if let q = qf, q.has(name) { return try q.anyFloats(name) }
            return try hf.floats(name).data
        }
        // A q4 matrix: from the quant file when it has codes for it, else quantized here.
        func q4(_ name: String) throws -> Q4Mat {
            if let q = qf, q.has(name + ".q") {
                let codes = try q.raw(name + ".q"), sc = try q.raw(name + ".scale"), bi = try q.raw(name + ".bias")
                let rows = codes.shape[0], cols = codes.shape[1] * 2
                guard sc.dtype == "F16", bi.dtype == "F16", codes.dtype == "U8" else { throw EngineError("\(name): bad quant dtypes") }
                let packed = Array(UnsafeBufferPointer(start: codes.ptr.assumingMemoryBound(to: UInt8.self), count: codes.count))
                let s16 = UnsafeBufferPointer(start: sc.ptr.assumingMemoryBound(to: Float16.self), count: sc.count / 2)
                let b16 = UnsafeBufferPointer(start: bi.ptr.assumingMemoryBound(to: Float16.self), count: bi.count / 2)
                var sb = [Float16](repeating: 0, count: 2 * s16.count)
                for i in 0..<s16.count { sb[2 * i] = s16[i]; sb[2 * i + 1] = b16[i] }
                return Q4Mat(packed: packed, sb: sb, rows: rows, cols: cols)
            }
            let (shape, data) = try hf.floats(name)
            let (p, b) = quantizeQ4Raw(data, rows: shape[0], cols: shape[1])
            return Q4Mat(packed: p, sb: b, rows: shape[0], cols: shape[1])
        }
        let st = hf
        let w = ModelWriter()
        let H = Dims.hidden, F = Dims.ff, V = Dims.vocab
        func emitQ4(_ id: Int, _ sbID: Int, _ m: Q4Mat) {
            w.add(id, m.packed.withUnsafeBufferPointer { Data(buffer: $0) })
            w.add(sbID, m.sb.withUnsafeBufferPointer { Data(buffer: $0) })
        }
        func emitF16(_ id: Int, _ sbID: Int, _ m: [Float]) {
            w.add(id, f16Data(m))
            w.add(sbID, Data())
        }
        let order = qkvRowOrder()
        var guOrder: [Int] = []  // stored row 4b+j of gate||up = original row
        for b in 0..<(F / 2) { guOrder += [2 * b, 2 * b + 1, F + 2 * b, F + 2 * b + 1] }

        if format == .q4a {
            try convertQ4A(hf: hf, qf: qf, writer: w, vec: vec, log: log)
            try w.write(to: output, format: format)
            return
        }
        let t0 = Date()
        if format == .q4 {
            emitQ4(TensorID.embed, TensorID.embedSB, try q4("model.embed_tokens.weight"))
        } else {
            emitF16(TensorID.embed, TensorID.embedSB, try st.floats("model.embed_tokens.weight").data)
        }
        _ = V
        w.add(TensorID.finalNorm, f32Data(try vec("model.norm.weight")))
        log(String(format: "embedding done (%.1fs)", Date().timeIntervalSince(t0)))

        for l in 0..<Dims.layers {
            let p = "model.layers.\(l)."
            w.add(TensorID.layer(l, TensorID.attnNorm), f32Data(try vec(p + "input_layernorm.weight")))
            w.add(TensorID.layer(l, TensorID.mlpNorm), f32Data(try vec(p + "post_attention_layernorm.weight")))
            let bias = try vec(p + "self_attn.q_proj.bias") + vec(p + "self_attn.k_proj.bias") + vec(p + "self_attn.v_proj.bias")
            w.add(TensorID.layer(l, TensorID.bqkv), f32Data(order.map { bias[$0] }))
            let names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"].map { p + $0 }
            if format == .q4 {
                let qkv = try q4(names[0]) + q4(names[1]) + q4(names[2])
                emitQ4(TensorID.layer(l, TensorID.wqkv), TensorID.layer(l, TensorID.wqkvSB), qkv.reordered(order))
                emitQ4(TensorID.layer(l, TensorID.wo), TensorID.layer(l, TensorID.woSB), try q4(p + "self_attn.o_proj.weight"))
                let gu = try q4(p + "mlp.gate_proj.weight") + q4(p + "mlp.up_proj.weight")
                emitQ4(TensorID.layer(l, TensorID.wgu), TensorID.layer(l, TensorID.wguSB), gu.reordered(guOrder))
                emitQ4(TensorID.layer(l, TensorID.wd), TensorID.layer(l, TensorID.wdSB), try q4(p + "mlp.down_proj.weight"))
            } else {
                let qkv = try names.flatMap { try st.floats($0).data }
                emitF16(TensorID.layer(l, TensorID.wqkv), TensorID.layer(l, TensorID.wqkvSB), reorderRows(qkv, cols: H, order: order))
                emitF16(TensorID.layer(l, TensorID.wo), TensorID.layer(l, TensorID.woSB), try st.floats(p + "self_attn.o_proj.weight").data)
                let gu = try st.floats(p + "mlp.gate_proj.weight").data + st.floats(p + "mlp.up_proj.weight").data
                emitF16(TensorID.layer(l, TensorID.wgu), TensorID.layer(l, TensorID.wguSB), reorderRows(gu, cols: H, order: guOrder))
                emitF16(TensorID.layer(l, TensorID.wd), TensorID.layer(l, TensorID.wdSB), try st.floats(p + "mlp.down_proj.weight").data)
            }
            log(String(format: "layer %d done (%.1fs)", l, Date().timeIntervalSince(t0)))
        }
        try w.write(to: output, format: format)
        log(String(format: "wrote %@ (%.1fs)", output, Date().timeIntervalSince(t0)))
    }

    static func convertQ4A(hf: SafeTensors, qf: SafeTensors?, writer w: ModelWriter, vec: (String) throws -> [Float],
                           log: (String) -> Void) throws {
        // Weights may come pre-transformed (folded activation scaling) from a quant file as f32.
        func mat(_ name: String) throws -> [Float] {
            if let q = qf, q.has(name) { return try q.anyFloats(name) }
            return try hf.floats(name).data
        }
        func emit(_ id: Int, _ sbID: Int, _ m: [Float], rows: Int, K: Int) {
            let packed = Q4A.pack(m, rows: rows, K: K)
            w.add(id, packed.withUnsafeBufferPointer { Data(buffer: $0) })
            w.add(sbID, Data())
        }
        let H = Dims.hidden, F = Dims.ff
        let t0 = Date()
        emit(TensorID.embed, TensorID.embedSB, try mat("model.embed_tokens.weight"), rows: Dims.vocab, K: H)
        w.add(TensorID.finalNorm, f32Data(try vec("model.norm.weight")))
        log(String(format: "embedding done (%.1fs)", Date().timeIntervalSince(t0)))
        let qo = Q4A.qkvOrder(), go = Q4A.guOrder(ff: F)
        for l in 0..<Dims.layers {
            let p = "model.layers.\(l)."
            w.add(TensorID.layer(l, TensorID.attnNorm), f32Data(try vec(p + "input_layernorm.weight")))
            w.add(TensorID.layer(l, TensorID.mlpNorm), f32Data(try vec(p + "post_attention_layernorm.weight")))
            let bias = try vec(p + "self_attn.q_proj.bias") + vec(p + "self_attn.k_proj.bias") + vec(p + "self_attn.v_proj.bias")
            w.add(TensorID.layer(l, TensorID.bqkv), f32Data(qo.map { bias[$0] }))
            let qkv = try mat(p + "self_attn.q_proj.weight") + mat(p + "self_attn.k_proj.weight") + mat(p + "self_attn.v_proj.weight")
            emit(TensorID.layer(l, TensorID.wqkv), TensorID.layer(l, TensorID.wqkvSB), reorderRows(qkv, cols: H, order: qo), rows: Dims.qkvRows, K: H)
            emit(TensorID.layer(l, TensorID.wo), TensorID.layer(l, TensorID.woSB), try mat(p + "self_attn.o_proj.weight"), rows: H, K: H)
            let gu = try mat(p + "mlp.gate_proj.weight") + mat(p + "mlp.up_proj.weight")
            emit(TensorID.layer(l, TensorID.wgu), TensorID.layer(l, TensorID.wguSB), reorderRows(gu, cols: H, order: go), rows: 2 * F, K: H)
            emit(TensorID.layer(l, TensorID.wd), TensorID.layer(l, TensorID.wdSB), try mat(p + "mlp.down_proj.weight"), rows: H, K: F)
            log(String(format: "layer %d done (%.1fs)", l, Date().timeIntervalSince(t0)))
        }
    }

    // Stored row i of the QKV matrix = original row qkvRowOrder()[i].
    static func qkvRowOrder() -> [Int] {
        var order: [Int] = []
        for b in 0..<36 {
            let head = b / 2, half = b % 2
            for i in 0..<16 { order.append(head * 64 + half * 16 + i) }
            for i in 0..<16 { order.append(head * 64 + 32 + half * 16 + i) }
        }
        return order
    }

    static func reorderRows(_ m: [Float], cols: Int, order: [Int]) -> [Float] {
        var out = [Float](repeating: 0, count: order.count * cols)
        for (i, r) in order.enumerated() { for c in 0..<cols { out[i * cols + c] = m[r * cols + c] } }
        return out
    }

    static func f32Data(_ v: [Float]) -> Data { v.withUnsafeBufferPointer { Data(buffer: $0) } }

    static func f16Data(_ v: [Float]) -> Data {
        var h = [Float16](repeating: 0, count: v.count)
        for i in 0..<v.count { h[i] = Float16(v[i]) }
        return h.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    // Affine 4-bit, group 64, f16 (scale, bias) per group: w ~ scale * q + bias, q in 0...15.
    // Per group, searches clipped ranges and refits (scale, bias) by least squares, keeping the
    // pair with the lowest squared error after rounding both to f16.
    static func quantizeQ4Raw(_ m: [Float], rows: Int, cols: Int) -> ([UInt8], [Float16]) {
        let groups = cols / 64
        var packed = [UInt8](repeating: 0, count: rows * cols / 2)
        var sb = [Float16](repeating: 0, count: rows * groups * 2)
        m.withUnsafeBufferPointer { src in
            packed.withUnsafeMutableBufferPointer { pk in
                sb.withUnsafeMutableBufferPointer { sbp in
                    let chunk = 64
                    DispatchQueue.concurrentPerform(iterations: (rows + chunk - 1) / chunk) { ci in
                        var q = [UInt8](repeating: 0, count: 64)
                        for r in (ci * chunk)..<min(rows, (ci + 1) * chunk) {
                            for g in 0..<groups {
                                let base = r * cols + g * 64
                                let (s, b) = quantGroup(src.baseAddress! + base, &q)
                                sbp[(r * groups + g) * 2] = s
                                sbp[(r * groups + g) * 2 + 1] = b
                                for j in 0..<32 { pk[base / 2 + j] = q[2 * j] | (q[2 * j + 1] << 4) }
                            }
                        }
                    }
                }
            }
        }
        return (packed, sb)
    }

    static let clips: [Float] = [0, 0.03, 0.06, 0.1, 0.15]

    @inline(__always)
    static func quantGroup(_ w: UnsafePointer<Float>, _ qOut: inout [UInt8]) -> (Float16, Float16) {
        var mn = w[0], mx = w[0]
        for i in 1..<64 { mn = min(mn, w[i]); mx = max(mx, w[i]) }
        if mx - mn < 1e-12 {
            for i in 0..<64 { qOut[i] = 0 }
            return (0, Float16(mn))
        }
        var bestErr = Float.infinity
        var bestS: Float16 = 0, bestB: Float16 = 0
        var q = [UInt8](repeating: 0, count: 64)
        let range = mx - mn
        for a in clips {
            for c in clips {
                let lo = mn + a * range, hi = mx - c * range
                if hi <= lo { continue }
                var s = (hi - lo) / 15, b = lo
                for iter in 0..<2 {
                    let inv = 1 / s
                    var sq: Float = 0, sw: Float = 0, sqq: Float = 0, sqw: Float = 0
                    for i in 0..<64 {
                        let v = min(15, max(0, ((w[i] - b) * inv).rounded()))
                        q[i] = UInt8(v)
                        sq += v; sw += w[i]; sqq += v * v; sqw += v * w[i]
                    }
                    if iter == 1 { break }
                    let den = 64 * sqq - sq * sq
                    if den <= 0 { break }
                    let s2 = (64 * sqw - sq * sw) / den
                    if s2 <= 0 { break }
                    s = s2
                    b = (sw - s2 * sq) / 64
                }
                // Round to f16 and requantize with the stored values.
                let sh = Float16(s), bh = Float16(b)
                let sf = Float(sh), bf = Float(bh)
                if sf <= 0 { continue }
                let inv = 1 / sf
                var err: Float = 0
                for i in 0..<64 {
                    let v = min(15, max(0, ((w[i] - bf) * inv).rounded()))
                    q[i] = UInt8(v)
                    let d = sf * v + bf - w[i]
                    err += d * d
                }
                if err < bestErr { bestErr = err; bestS = sh; bestB = bh; qOut = q }
            }
        }
        return (bestS, bestB)
    }
}
