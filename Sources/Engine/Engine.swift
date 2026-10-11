import CShim
import Foundation
import Metal

// The inference engine. One request = one prefill pass over the tokens not already in the KV cache,
// then greedy decode steps that run entirely on the GPU: lm_head writes per-threadgroup maxima, the
// commit kernel picks the token, records it in a shared buffer and embeds it for the next step. The
// CPU only encodes ahead and polls the shared token buffer, so there is no CPU round trip per token.

public final class Engine {
    public let device: MTLDevice
    public let model: ModelFile
    public let format: WeightFormat
    public let maxContext: Int
    public let maxPass: Int
    public let maxSteps: Int
    public var eos: (Int32, Int32) = (151645, 151643)

    let queue: MTLCommandQueue
    let wbuf: MTLBuffer
    let x, xn, q, attn, hbuf, kc, vc, state, promptToks, outToks, pval, pidx, logits, rope: MTLBuffer
    let xnh, attnh, hh: MTLBuffer
    let drafts, promptRecent: MTLBuffer
    var ngram: MTLBuffer
    let pEmbedPrompt, pCommit, pAttnIn, pAttention, pOProj, pMlpIn, pMlpOut, pLMHead: MTLComputePipelineState
    let pRmsRowsH, pF32ToH, pNoop: MTLComputePipelineState
    var pSpin: MTLComputePipelineState?
    var pCommitSpec: MTLComputePipelineState?
    var pLmhSketch: [Int: MTLComputePipelineState] = [:], pLmhExact: [Int: MTLComputePipelineState] = [:]
    var pLmhTau: MTLComputePipelineState?
    let lmhBounds, lmhTau: MTLBuffer
    /// Two-stage lm_head (2-bit sketch + exact recompute of candidate groups); needs the sketch plane.
    /// Off by default: ~4% faster decode, but its argmax equals the full lm_head's only with
    /// overwhelming probability (a residual beyond K sigma would flip a near-tie), not by construction.
    public var useSketchHead = ProcessInfo.processInfo.environment["ILL_SKETCH"] != nil
    public var sketchK: Float = Float(ProcessInfo.processInfo.environment["ILL_SKETCH_K"] ?? "") ?? 6
    var pSpec: [Int: [MTLComputePipelineState]] = [:]   // T -> [attn_in, o_proj, mlp_in, mlp_out, lm_head]
    var pSpecVar: [String: [MTLComputePipelineState]] = [:]   // exploration variants ("2r4", "3r8")
    static func specRows(_ T: Int) -> Int { 8 }
    /// Rows per speculative pass (1 = off): the last token plus up to specT-1 n-gram drafts.
    public var specT = 1
    /// Best measured speculation width per format (verify cost vs accepted drafts).
    public var defaultSpecT: Int { format == .q4a ? 1 : 3 }
    var pG2: [[MTLComputePipelineState]] = []  // [TF-1][qkv, o, gu, down]
    var pG3: [Int: [MTLComputePipelineState]] = [:]  // tokens per tile (8, 16, 32) -> [qkv, o, gu, down]
    var pG3Lmh: [Int: MTLComputePipelineState] = [:] // tokens per tile (8, 16) -> lm_head with argmax partials
    /// Verify passes on the Neural Accelerators (prefill GEMMs, all rows, MPP lm_head): rows are
    /// nearly free there, so a pass can check up to 15 drafts. specT is then the rows per pass.
    public var wideVerify = ProcessInfo.processInfo.environment["ILL_WIDE"] != nil
    public var useGemm3 = ProcessInfo.processInfo.environment["ILL_NO_G3"] == nil
    /// Prompts of at least this many new tokens use the GEMM prefill path.
    public var gemmThreshold = 2
    var inflight: [MTLCommandBuffer] = []
    var library: MTLLibrary?
    var q4a: Bool { format == .q4a }
    /// q4 and q8 share the T-templated decode/verify kernels.
    var tFamily: Bool { format != .q4a }
    var residency: MTLResidencySet?

    /// Tokens whose keys/values are valid in the cache at positions 0..<count.
    public private(set) var cachedTokens: [Int32] = []

    public init(modelPath: String, maxContext: Int = 2048, maxPass: Int = 512, maxSteps: Int = 4096,
                copyWeights: Bool = false, pinMemory: Bool = true) throws {
        let tI0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var marks: [(String, UInt64)] = []
        func mark(_ s: String) { marks.append((s, clock_gettime_nsec_np(CLOCK_UPTIME_RAW))) }
        // MTLCopyAllDevices skips the display-GPU bookkeeping of MTLCreateSystemDefaultDevice:
        // 12-16 ms instead of ~40 ms in a fresh process (Apple Silicon has a single GPU anyway).
        guard let dev = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice() else { throw EngineError("no Metal device") }
        mark("device")
        device = dev
        model = try ModelFile(path: modelPath)
        format = model.format
        mark("mmap")
        self.maxContext = maxContext
        self.maxPass = maxPass
        self.maxSteps = maxSteps
        queue = dev.makeCommandQueue()!

        if copyWeights {
            // Anonymous (not file-backed) copy: survives long idle periods without re-faulting.
            guard let wb = dev.makeBuffer(bytes: model.base, length: model.length,
                                          options: [.storageModeShared, .hazardTrackingModeUntracked])
            else { throw EngineError("cannot allocate the weight buffer") }
            wbuf = wb
        } else {
            guard let wb = dev.makeBuffer(bytesNoCopy: model.base, length: model.length,
                                          options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil)
            else { throw EngineError("cannot wrap model weights in a Metal buffer") }
            wbuf = wb
        }

        mark("weights buffer")
        func buf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: max(bytes, 16), options: .storageModeShared)! }
        let H = Dims.hidden
        x = buf(maxPass * H * 4)
        xn = buf(maxPass * H * 4)
        xnh = buf((maxPass + 32) * H * 2)
        attnh = buf((maxPass + 32) * H * 2)
        hh = buf((maxPass + 32) * Dims.ff * 2)
        q = buf(maxPass * H * 4)
        attn = buf(maxPass * H * 4)
        hbuf = buf(maxPass * Dims.ff * 4)
        let kvBytes = Dims.layers * Dims.kvHeads * maxContext * Dims.headDim * 2
        kc = buf(kvBytes)
        vc = buf(kvBytes)
        state = buf(64)
        promptToks = buf(maxPass * 4)
        outToks = buf(maxSteps * 4)
        pval = buf(16 * Dims.vocab / 32 * 4)   // lm_head argmax partials [row][VOCAB / 32], up to 16 rows
        pidx = buf(16 * Dims.vocab / 32 * 4)
        drafts = buf(64)
        lmhBounds = buf(4 * 2 * Dims.vocab / 32 * 4)
        lmhTau = buf(64)
        promptRecent = buf(4 * (4 + 256))   // [count, -, -, -, last <= 256 prompt tokens] for prompt lookup
        ngram = buf(16)
        logits = buf(Dims.vocab * 4)
        rope = buf(maxContext * 32 * 8)
        mark("activation buffers")
        let r = rope.contents().assumingMemoryBound(to: Float.self)
        for p in 0..<maxContext {
            for i in 0..<32 {
                let invf = Float(1.0 / pow(Dims.ropeTheta, Double(2 * i) / 64.0))
                let a = Double(Float(p) * invf)
                r[(p * 32 + i) * 2] = Float(cos(a))
                r[(p * 32 + i) * 2 + 1] = Float(sin(a))
            }
        }

        mark("rope table")
        // Keep every allocation resident: without this, an idle process loses its GPU mappings
        // and the next token page-faults the weights back in (tens of ms per token).
        let rd = MTLResidencySetDescriptor()
        rd.initialCapacity = 32
        if pinMemory, ProcessInfo.processInfo.environment["ILL_NO_RESIDENCY"] == nil, let rs = try? dev.makeResidencySet(descriptor: rd) {
            for b in [wbuf, x, xn, xnh, attnh, hh, q, attn, hbuf, kc, vc, state, promptToks, outToks, pval, pidx, logits, rope, drafts, promptRecent, lmhBounds, lmhTau] {
                rs.addAllocation(b)
            }
            rs.commit()
            rs.requestResidency()
            queue.addResidencySet(rs)
            residency = rs
        }

        mark("residency")
        let opts = MTLCompileOptions()
        var macros: [String: NSObject] = ["FMT": NSNumber(value: model.format.rawValue)]
        // Experiments: ILL_DEFINES="A B=2" adds preprocessor macros to the shader compile.
        for d in (ProcessInfo.processInfo.environment["ILL_DEFINES"] ?? "").split(separator: " ") {
            let kv = d.split(separator: "=", maxSplits: 1)
            macros[String(kv[0])] = NSNumber(value: kv.count > 1 ? Int(kv[1]) ?? 1 : 1)
        }
        opts.preprocessorMacros = macros
        opts.mathMode = ProcessInfo.processInfo.environment["ILL_MATH"] == "safe" ? .safe : (ProcessInfo.processInfo.environment["ILL_MATH"] == "relaxed" ? .relaxed : .fast)
        opts.languageVersion = .version4_0
        let tLib0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let lib = try dev.makeLibrary(source: shaderSource, options: opts)
        library = lib
        let tLib1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        func pipe(_ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw EngineError("missing kernel \(name)") }
            return try dev.makeComputePipelineState(function: f)
        }
        pEmbedPrompt = try pipe("k_embed_prompt")
        pCommit = try pipe("k_commit")
        pAttnIn = try pipe("k_attn_in")
        pAttention = try pipe("k_attention")
        pOProj = try pipe("k_o_proj")
        pMlpIn = try pipe("k_mlp_in")
        pMlpOut = try pipe("k_mlp_out")
        pLMHead = try pipe("k_lm_head")
        pRmsRowsH = try pipe("k_rmsnorm_rows_h")
        pSpin = try? pipe("k_spin")
        pF32ToH = try pipe("k_f32_to_h")
        pNoop = try pipe("k_noop")
        if model.format == .q4 && model.size(TensorID.embedHi) > 0 {
            for t in 1...4 { pLmhSketch[t] = try pipe("k_lmh_sketch_\(t)"); pLmhExact[t] = try pipe("k_lmh_exact_\(t)") }
            pLmhTau = try pipe("k_lmh_tau")
        } else {
            useSketchHead = false
        }
        if model.format != .q4a {
            pCommitSpec = try pipe("k_commit_spec")
            for t in 2...4 { pSpec[t] = try ["k_attn_in_", "k_o_proj_", "k_mlp_in_", "k_mlp_out_", "k_lm_head_"].map { try pipe($0 + "\(t)") } }
        }
        if ProcessInfo.processInfo.environment["ILL_PIPE_INFO"] != nil {
            var all: [(String, MTLComputePipelineState)] = [("attn_in", pAttnIn), ("o_proj", pOProj), ("mlp_in", pMlpIn), ("mlp_out", pMlpOut), ("lm_head", pLMHead)]
            for (t, ps) in pSpec.sorted(by: { $0.key < $1.key }) {
                for (i, nm) in ["attn_in", "o_proj", "mlp_in", "mlp_out", "lm_head"].enumerated() { all.append(("\(nm) T=\(t)", ps[i])) }
            }
            for (nm, p) in all {
                FileHandle.standardError.write("\(nm): maxThreads \(p.maxTotalThreadsPerThreadgroup) tgmem \(p.staticThreadgroupMemoryLength)\n".data(using: .utf8)!)
            }
        }
        if ProcessInfo.processInfo.environment["ILL_VERBOSE_INIT"] != nil {
            let tP = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var prev = tI0
            var line = "init:"
            for (n, t) in marks { line += String(format: " %@ %.1f", n, Double(t - prev) / 1e6); prev = t }
            line += String(format: " library %.1f pipelines %.1f ms\n", Double(tLib1 - tLib0) / 1e6, Double(tP - tLib1) / 1e6)
            FileHandle.standardError.write(line.data(using: .utf8)!)
        }
        if model.format != .q4a {
            for tf in 1...4 {
                pG2.append(try ["qkv", "o", "gu", "down"].map { try pipe("k_g2_\($0)_\(tf)") })
            }
        }
        do {
            for tt in [8, 16, 24, 32] { pG3[tt] = try ["qkv", "o", "gu", "down"].map { try pipe("k_g3_\($0)_\(tt)") } }
            for tt in [8, 16] { pG3Lmh[tt] = try pipe("k_g3_lmh_\(tt)") }
        }
    }

    // MARK: encoding

    @inline(__always) private func w(_ e: MTLComputeCommandEncoder, _ id: Int, _ i: Int) {
        e.setBuffer(wbuf, offset: model.offset(id), index: i)
    }

    @inline(__always) private func grid(_ e: MTLComputeCommandEncoder, _ x: Int, _ y: Int, _ threads: Int = 256) {
        e.dispatchThreadgroups(MTLSize(width: x, height: y, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    private func encodeLayer(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int) {
        for part in 0..<5 { encodeLayerPart(e, l, part: part, n: n) }
    }

    private func encodeLayerPart(_ e: MTLComputeCommandEncoder, _ l: Int, part: Int, n: Int = 1) {
        let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
        if part == 0 {
        e.setComputePipelineState(pAttnIn)
        w(e, TensorID.layer(l, TensorID.wqkv), 0)
        w(e, TensorID.layer(l, TensorID.wqkvSB), 1)
        w(e, TensorID.layer(l, TensorID.bqkv), 2)
        w(e, TensorID.layer(l, TensorID.attnNorm), 3)
        e.setBuffer(x, offset: 0, index: 4)
        e.setBuffer(q, offset: 0, index: 5)
        e.setBuffer(kc, offset: kvOff, index: 6)
        e.setBuffer(vc, offset: kvOff, index: 7)
        e.setBuffer(rope, offset: 0, index: 8)
        e.setBuffer(state, offset: 0, index: 9)
        if q4a { grid(e, 36, n, 64) } else if tFamily { grid(e, 36, n, 128) } else { grid(e, 36, n) }
        } else if part == 1 {
        e.setComputePipelineState(pAttention)
        e.setBuffer(q, offset: 0, index: 0)
        e.setBuffer(kc, offset: kvOff, index: 1)
        e.setBuffer(vc, offset: kvOff, index: 2)
        e.setBuffer(attn, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        var toff: Int32 = 0
        e.setBytes(&toff, length: 4, index: 5)
        grid(e, Dims.heads, n)
        } else if part == 2 {
        e.setComputePipelineState(pOProj)
        w(e, TensorID.layer(l, TensorID.wo), 0)
        w(e, TensorID.layer(l, TensorID.woSB), 1)
        e.setBuffer(attn, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        if q4a { grid(e, 28, n, 64) } else if tFamily { grid(e, 56, n, 64) } else { grid(e, Dims.hidden / 32, n) }
        } else if part == 3 {
        e.setComputePipelineState(pMlpIn)
        w(e, TensorID.layer(l, TensorID.wgu), 0)
        w(e, TensorID.layer(l, TensorID.wguSB), 1)
        w(e, TensorID.layer(l, TensorID.mlpNorm), 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(hbuf, offset: 0, index: 4)
        e.setBuffer(state, offset: 0, index: 5)
        if q4a { grid(e, 304, n, 64) } else if tFamily { grid(e, 608, n, 64) } else { grid(e, Dims.ff / 16, n) }
        } else {
        e.setComputePipelineState(pMlpOut)
        w(e, TensorID.layer(l, TensorID.wd), 0)
        w(e, TensorID.layer(l, TensorID.wdSB), 1)
        e.setBuffer(hbuf, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        if q4a { grid(e, 56, n, 128) } else if tFamily { grid(e, 56, n, 64) } else { grid(e, Dims.hidden / 32, n) }
        }
    }

    private func encodeG2(_ e: MTLComputeCommandEncoder, _ kind: Int, a: MTLBuffer, wID: Int, sbID: Int, biasID: Int?,
                          outf: MTLBuffer, outh: MTLBuffer, kvOff: Int, rows: Int, n: Int) {
        if useGemm3 {
            let tt = n <= 8 ? 8 : (n <= 16 ? 16 : (n <= 24 ? 24 : 32))
            e.setComputePipelineState(pG3[tt]![kind])
            e.setBuffer(a, offset: 0, index: 0)
            w(e, wID, 1)
            w(e, sbID, 2)
            w(e, biasID ?? sbID, 3)
            e.setBuffer(outf, offset: 0, index: 4)
            e.setBuffer(outh, offset: 0, index: 5)
            e.setBuffer(kc, offset: kvOff, index: 6)
            e.setBuffer(vc, offset: kvOff, index: 7)
            e.setBuffer(rope, offset: 0, index: 8)
            e.setBuffer(state, offset: 0, index: 9)
            grid(e, rows / 32, (n + tt - 1) / tt, 128)
            return
        }
        let tf = min(4, (n + 7) / 8)
        e.setComputePipelineState(pG2[tf - 1][kind])
        e.setBuffer(a, offset: 0, index: 0)
        w(e, wID, 1)
        w(e, sbID, 2)
        w(e, biasID ?? sbID, 3)
        e.setBuffer(outf, offset: 0, index: 4)
        e.setBuffer(outh, offset: 0, index: 5)
        e.setBuffer(kc, offset: kvOff, index: 6)
        e.setBuffer(vc, offset: kvOff, index: 7)
        e.setBuffer(rope, offset: 0, index: 8)
        e.setBuffer(state, offset: 0, index: 9)
        grid(e, rows / 32, (n + tf * 8 - 1) / (tf * 8), kind == 3 ? 256 : 128)
    }

    /// O, gate/up and down for the single row n-1 with the decode kernels (buffer offsets select the row).
    private func encodeLastRowTail(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int, kvOff: Int) {
        let rowF = (n - 1) * Dims.hidden * 4
        e.setComputePipelineState(pOProj)
        w(e, TensorID.layer(l, TensorID.wo), 0)
        w(e, TensorID.layer(l, TensorID.woSB), 1)
        e.setBuffer(attn, offset: rowF, index: 2)
        e.setBuffer(x, offset: rowF, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        if q4a { grid(e, 28, 1, 64) } else if tFamily { grid(e, 56, 1, 64) } else { grid(e, Dims.hidden / 32, 1) }
        e.setComputePipelineState(pMlpIn)
        w(e, TensorID.layer(l, TensorID.wgu), 0)
        w(e, TensorID.layer(l, TensorID.wguSB), 1)
        w(e, TensorID.layer(l, TensorID.mlpNorm), 2)
        e.setBuffer(x, offset: rowF, index: 3)
        e.setBuffer(hbuf, offset: 0, index: 4)
        e.setBuffer(state, offset: 0, index: 5)
        if q4a { grid(e, 304, 1, 64) } else if tFamily { grid(e, 608, 1, 64) } else { grid(e, Dims.ff / 16, 1) }
        e.setComputePipelineState(pMlpOut)
        w(e, TensorID.layer(l, TensorID.wd), 0)
        w(e, TensorID.layer(l, TensorID.wdSB), 1)
        e.setBuffer(hbuf, offset: 0, index: 2)
        e.setBuffer(x, offset: rowF, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        if q4a { grid(e, 56, 1, 128) } else if tFamily { grid(e, 56, 1, 64) } else { grid(e, Dims.hidden / 32, 1) }
    }

    private func encodePrefillLayer2(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int, allRows: Bool = false) {
        let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
        let L = { TensorID.layer(l, $0) }
        e.setComputePipelineState(pRmsRowsH)
        e.setBuffer(x, offset: 0, index: 0); w(e, L(TensorID.attnNorm), 1); e.setBuffer(xnh, offset: 0, index: 2); e.setBuffer(state, offset: 0, index: 3)
        grid(e, n, 1)
        encodeG2(e, 0, a: xnh, wID: L(TensorID.wqkv), sbID: L(TensorID.wqkvSB), biasID: L(TensorID.bqkv), outf: q, outh: hh, kvOff: kvOff, rows: Dims.qkvRows, n: n)
        e.setComputePipelineState(pAttention)
        e.setBuffer(q, offset: 0, index: 0); e.setBuffer(kc, offset: kvOff, index: 1); e.setBuffer(vc, offset: kvOff, index: 2)
        e.setBuffer(attn, offset: 0, index: 3); e.setBuffer(state, offset: 0, index: 4)
        if l == Dims.layers - 1 && n > 1 && !allRows {
            // Last layer: only the last token's output is needed (the KV of all tokens is already
            // written by the QKV kernel), so attention, O and the MLP run for that token alone,
            // with the decode kernels on row n-1.
            var toff = Int32(n - 1)
            e.setBytes(&toff, length: 4, index: 5)
            grid(e, Dims.heads, 1)
            encodeLastRowTail(e, l, n: n, kvOff: kvOff)
            return
        }
        var toff: Int32 = 0
        e.setBytes(&toff, length: 4, index: 5)
        grid(e, Dims.heads, n)
        e.setComputePipelineState(pF32ToH)
        e.setBuffer(attn, offset: 0, index: 0); e.setBuffer(attnh, offset: 0, index: 1); e.setBuffer(state, offset: 0, index: 2)
        e.dispatchThreads(MTLSize(width: n * Dims.hidden, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encodeG2(e, 1, a: attnh, wID: L(TensorID.wo), sbID: L(TensorID.woSB), biasID: nil, outf: x, outh: hh, kvOff: kvOff, rows: Dims.hidden, n: n)
        e.setComputePipelineState(pRmsRowsH)
        e.setBuffer(x, offset: 0, index: 0); w(e, L(TensorID.mlpNorm), 1); e.setBuffer(xnh, offset: 0, index: 2); e.setBuffer(state, offset: 0, index: 3)
        grid(e, n, 1)
        encodeG2(e, 2, a: xnh, wID: L(TensorID.wgu), sbID: L(TensorID.wguSB), biasID: nil, outf: x, outh: hh, kvOff: kvOff, rows: 2 * Dims.ff, n: n)
        encodeG2(e, 3, a: hh, wID: L(TensorID.wd), sbID: L(TensorID.wdSB), biasID: nil, outf: x, outh: hh, kvOff: kvOff, rows: Dims.hidden, n: n)
    }

    /// Sketch + threshold + exact-candidates lm_head over the last T rows of x (writes the same partials).
    private func encodeSketchHead(_ e: MTLComputeCommandEncoder, T: Int) {
        e.setComputePipelineState(pLmhSketch[T]!)
        w(e, TensorID.embedHi, 0); w(e, TensorID.embedSB, 1); w(e, TensorID.finalNorm, 2)
        e.setBuffer(x, offset: 0, index: 3); e.setBuffer(lmhBounds, offset: 0, index: 4); e.setBuffer(state, offset: 0, index: 5)
        var k = sketchK
        e.setBytes(&k, length: 4, index: 6)
        grid(e, Dims.vocab / 32, 1, 128)
        e.setComputePipelineState(pLmhTau!)
        e.setBuffer(lmhBounds, offset: 0, index: 0); e.setBuffer(lmhTau, offset: 0, index: 1); e.setBuffer(state, offset: 0, index: 2)
        var t32 = Int32(T)
        e.setBytes(&t32, length: 4, index: 3)
        grid(e, 1, 1, 1024)
        e.setComputePipelineState(pLmhExact[T]!)
        w(e, TensorID.embed, 0); w(e, TensorID.embedSB, 1); w(e, TensorID.finalNorm, 2)
        e.setBuffer(x, offset: 0, index: 3); e.setBuffer(pval, offset: 0, index: 4); e.setBuffer(pidx, offset: 0, index: 5)
        e.setBuffer(state, offset: 0, index: 6); e.setBuffer(lmhBounds, offset: 0, index: 7); e.setBuffer(lmhTau, offset: 0, index: 8)
        grid(e, Dims.vocab / 32, 1, 128)
    }

    private func encodeLMHead(_ e: MTLComputeCommandEncoder, writeLogits: Bool) {
        if useSketchHead && !writeLogits { encodeSketchHead(e, T: 1); return }
        e.setComputePipelineState(pLMHead)
        w(e, TensorID.embed, 0)
        w(e, TensorID.embedSB, 1)
        w(e, TensorID.finalNorm, 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(pval, offset: 0, index: 4)
        e.setBuffer(pidx, offset: 0, index: 5)
        e.setBuffer(state, offset: 0, index: 6)
        e.setBuffer(logits, offset: 0, index: 7)
        var wl: Int32 = writeLogits ? 1 : 0
        e.setBytes(&wl, length: 4, index: 8)
        grid(e, Dims.vocab / 32, 1, q4a ? 64 : (tFamily ? 128 : 256))
    }

    private func encodeCommit(_ e: MTLComputeCommandEncoder) {
        e.setComputePipelineState(pCommit)
        e.setBuffer(pval, offset: 0, index: 0)
        e.setBuffer(pidx, offset: 0, index: 1)
        e.setBuffer(state, offset: 0, index: 2)
        e.setBuffer(outToks, offset: 0, index: 3)
        w(e, TensorID.embed, 4)
        w(e, TensorID.embedSB, 5)
        e.setBuffer(x, offset: 0, index: 6)
        grid(e, 1, 1, 1024)
    }

    private func encodeCommitSpec(_ e: MTLComputeCommandEncoder) {
        e.setComputePipelineState(pCommitSpec!)
        e.setBuffer(pval, offset: 0, index: 0)
        e.setBuffer(pidx, offset: 0, index: 1)
        e.setBuffer(state, offset: 0, index: 2)
        e.setBuffer(outToks, offset: 0, index: 3)
        w(e, TensorID.embed, 4)
        w(e, TensorID.embedSB, 5)
        e.setBuffer(x, offset: 0, index: 6)
        e.setBuffer(drafts, offset: 0, index: 7)
        e.setBuffer(ngram, offset: 0, index: 8)
        e.setBuffer(promptRecent, offset: 0, index: 9)
        grid(e, 1, 1, 1024)
    }

    /// One speculative pass over T rows (last token + drafts): every weight read once for all rows.
    private func encodeSpecPass(_ e: MTLComputeCommandEncoder, T: Int, p pv: [MTLComputePipelineState]? = nil, R rv: Int? = nil) {
        let p = pv ?? pSpec[T]!
        let R = rv ?? Engine.specRows(T)
        for l in 0..<Dims.layers {
            let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
            let L = { TensorID.layer(l, $0) }
            e.setComputePipelineState(p[0])
            w(e, L(TensorID.wqkv), 0); w(e, L(TensorID.wqkvSB), 1); w(e, L(TensorID.bqkv), 2); w(e, L(TensorID.attnNorm), 3)
            e.setBuffer(x, offset: 0, index: 4); e.setBuffer(q, offset: 0, index: 5)
            e.setBuffer(kc, offset: kvOff, index: 6); e.setBuffer(vc, offset: kvOff, index: 7)
            e.setBuffer(rope, offset: 0, index: 8); e.setBuffer(state, offset: 0, index: 9)
            grid(e, 36, 1, 32 * (32 / R))
            e.setComputePipelineState(pAttention)
            e.setBuffer(q, offset: 0, index: 0); e.setBuffer(kc, offset: kvOff, index: 1); e.setBuffer(vc, offset: kvOff, index: 2)
            e.setBuffer(attn, offset: 0, index: 3); e.setBuffer(state, offset: 0, index: 4)
            var toff: Int32 = 0
            e.setBytes(&toff, length: 4, index: 5)
            grid(e, Dims.heads, T)
            e.setComputePipelineState(p[1])
            w(e, L(TensorID.wo), 0); w(e, L(TensorID.woSB), 1)
            e.setBuffer(attn, offset: 0, index: 2); e.setBuffer(x, offset: 0, index: 3); e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.hidden / (2 * R), 1, 64)
            e.setComputePipelineState(p[2])
            w(e, L(TensorID.wgu), 0); w(e, L(TensorID.wguSB), 1); w(e, L(TensorID.mlpNorm), 2)
            e.setBuffer(x, offset: 0, index: 3); e.setBuffer(hbuf, offset: 0, index: 4); e.setBuffer(state, offset: 0, index: 5)
            grid(e, 2 * Dims.ff / (2 * R), 1, 64)
            e.setComputePipelineState(p[3])
            w(e, L(TensorID.wd), 0); w(e, L(TensorID.wdSB), 1)
            e.setBuffer(hbuf, offset: 0, index: 2); e.setBuffer(x, offset: 0, index: 3); e.setBuffer(state, offset: 0, index: 4)
            grid(e, Dims.hidden / (2 * R), 1, 64)
        }
        if useSketchHead {
            encodeSketchHead(e, T: T)
        } else {
            e.setComputePipelineState(p[4])
            w(e, TensorID.embed, 0); w(e, TensorID.embedSB, 1); w(e, TensorID.finalNorm, 2)
            e.setBuffer(x, offset: 0, index: 3); e.setBuffer(pval, offset: 0, index: 4); e.setBuffer(pidx, offset: 0, index: 5)
            e.setBuffer(state, offset: 0, index: 6); e.setBuffer(logits, offset: 0, index: 7)
            var wl: Int32 = 0
            e.setBytes(&wl, length: 4, index: 8)
            grid(e, Dims.vocab / 32, 1, 32 * (32 / R))
        }
        encodeCommitSpec(e)
    }

    /// Verify pass over T rows (last token + drafts) on the prefill GEMMs, then the MPP lm_head.
    private func encodeWidePass(_ e: MTLComputeCommandEncoder, T: Int, commit: Bool = true) {
        for l in 0..<Dims.layers { encodePrefillLayer2(e, l, n: T, allRows: true) }
        e.setComputePipelineState(pRmsRowsH)
        e.setBuffer(x, offset: 0, index: 0); w(e, TensorID.finalNorm, 1); e.setBuffer(xnh, offset: 0, index: 2); e.setBuffer(state, offset: 0, index: 3)
        grid(e, T, 1)
        e.setComputePipelineState(pG3Lmh[T <= 8 ? 8 : 16]!)
        e.setBuffer(xnh, offset: 0, index: 0); w(e, TensorID.embed, 1); w(e, TensorID.embedSB, 2)
        e.setBuffer(pval, offset: 0, index: 3); e.setBuffer(pidx, offset: 0, index: 4); e.setBuffer(state, offset: 0, index: 5)
        grid(e, Dims.vocab / 32, 1, 128)
        if commit { encodeCommitSpec(e) }
    }

    private func encodeSpecPassNoCommit(_ e: MTLComputeCommandEncoder, T: Int, p: [MTLComputePipelineState], R: Int) {
        let saved = pCommitSpec
        pCommitSpec = pNoop
        encodeSpecPass(e, T: T, p: p, R: R)
        pCommitSpec = saved
    }

    private func encodePrefill(_ e: MTLComputeCommandEncoder, n: Int, writeLogits: Bool) {
        e.setComputePipelineState(pEmbedPrompt)
        w(e, TensorID.embed, 0)
        w(e, TensorID.embedSB, 1)
        e.setBuffer(promptToks, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, n, 1)
        for l in 0..<Dims.layers {
            if n >= gemmThreshold { encodePrefillLayer2(e, l, n: n) } else { encodeLayer(e, l, n: n) }
        }
        encodeLMHead(e, writeLogits: writeLogits)
        if specActive { encodeCommitSpec(e) } else { encodeCommit(e) }
    }

    private var specActive = false

    private func encodeDecodeStep(_ e: MTLComputeCommandEncoder, writeLogits: Bool) {
        for l in 0..<Dims.layers { encodeLayer(e, l, n: 1) }
        encodeLMHead(e, writeLogits: writeLogits)
        encodeCommit(e)
    }

    private func newEncoder() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let cb = queue.makeCommandBufferWithUnretainedReferences()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
        return (cb, e)
    }

    // MARK: profiling

    /// GPU time (ms, median of `iters`) of command buffers encoded by `body`, with the state set to
    /// a pass of `n` tokens at position `pos`.
    public func gpuTime(n: Int, pos: Int, iters: Int = 20, _ body: (MTLComputeCommandEncoder) -> Void) -> Double {
        waitIdle()
        var times: [Double] = []
        for _ in 0..<iters {
            let s = state.contents().assumingMemoryBound(to: Int32.self)
            s[0] = Int32(pos); s[1] = Int32(n); s[2] = 0; s[3] = 0
            s[4] = Int32(maxContext); s[5] = Int32(maxSteps); s[6] = -5; s[7] = -5
            let (cb, e) = newEncoder()
            body(e)
            e.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }
        times.sort()
        return times[times.count / 2]
    }

    public func profile(n: Int, pos: Int) -> [(String, Double)] {
        var r: [(String, Double)] = []
        let pt = promptToks.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<maxPass { pt[i] = Int32(1000 + i) }
        if gemmThreshold != Int.max {
        r.append(("prefill pass (n=\(n)) incl lm_head+commit", gpuTime(n: n, pos: pos) { e in self.encodePrefill(e, n: n, writeLogits: false) }))
        let kv = { (l: Int) in l * Dims.kvHeads * self.maxContext * Dims.headDim * 2 }
        r.append(("  24x g2 qkv", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 0, a: self.xnh, wID: TensorID.layer(l, TensorID.wqkv), sbID: TensorID.layer(l, TensorID.wqkvSB), biasID: TensorID.layer(l, TensorID.bqkv), outf: self.q, outh: self.hh, kvOff: kv(l), rows: Dims.qkvRows, n: n) } }))
        r.append(("  24x attention", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers {
                e.setComputePipelineState(self.pAttention); e.setBuffer(self.q, offset: 0, index: 0); e.setBuffer(self.kc, offset: kv(l), index: 1)
                e.setBuffer(self.vc, offset: kv(l), index: 2); e.setBuffer(self.attn, offset: 0, index: 3); e.setBuffer(self.state, offset: 0, index: 4)
                var toff: Int32 = 0; e.setBytes(&toff, length: 4, index: 5)
                self.grid(e, Dims.heads, n) } }))
        r.append(("  24x g2 o", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 1, a: self.attnh, wID: TensorID.layer(l, TensorID.wo), sbID: TensorID.layer(l, TensorID.woSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: Dims.hidden, n: n) } }))
        r.append(("  24x g2 gu", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 2, a: self.xnh, wID: TensorID.layer(l, TensorID.wgu), sbID: TensorID.layer(l, TensorID.wguSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: 2 * Dims.ff, n: n) } }))
        r.append(("  24x g2 down", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 3, a: self.hh, wID: TensorID.layer(l, TensorID.wd), sbID: TensorID.layer(l, TensorID.wdSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: Dims.hidden, n: n) } }))
        r.append(("  lm_head + commit", gpuTime(n: n, pos: pos) { e in self.encodeLMHead(e, writeLogits: false); self.encodeCommit(e) }))
        }
        r.append(("decode step (1 token)", gpuTime(n: 1, pos: pos + n) { e in self.encodeDecodeStep(e, writeLogits: false) }))
        r.append(("decode 16 steps / 16", gpuTime(n: 1, pos: pos + n, iters: 5) { e in for _ in 0..<16 { self.encodeDecodeStep(e, writeLogits: false) } } / 16))
        if tFamily {
            let s0 = state.contents().assumingMemoryBound(to: Int32.self)
            var variants: [(String, Int, [MTLComputePipelineState], Int)] = [("T=1", 1, [pAttnIn, pOProj, pMlpIn, pMlpOut, pLMHead], 8)]
            for T in 2...4 { if let p = pSpec[T] { variants.append(("T=\(T)", T, p, Engine.specRows(T))) } }
            if let p = pSpecVar["2r4"] { variants.append(("T=2 R=4", 2, p, 4)) }
            if pG3Lmh[8] != nil {
                for T in [2, 4, 8, 12, 16] {
                    r.append(("wide pass T=\(T) (MPP layers + lm_head)", gpuTime(n: T, pos: pos + n) { e in s0[9] = Int32(T); self.encodeWidePass(e, T: T, commit: false) }))
                }
            }
            if let p = pSpecVar["3r8"] { variants.append(("T=3 R=8", 3, p, 8)) }
            for (name, T, p, R) in variants {
                r.append(("pass \(name) (layers + lm_head)", gpuTime(n: T, pos: pos + n) { e in
                    s0[9] = Int32(T)
                    self.encodeSpecPassNoCommit(e, T: T, p: p, R: R)
                }))
            }
        }
        r.append(("  24x attn_in (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 0) } }))
        r.append(("  24x attention (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 1) } }))
        r.append(("  24x o_proj (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 2) } }))
        r.append(("  24x mlp_in (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 3) } }))
        r.append(("  24x mlp_out (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 4) } }))
        r.append(("  lm_head (decode)", gpuTime(n: 1, pos: pos + n) { e in self.encodeLMHead(e, writeLogits: false) }))
        r.append(("  commit (decode)", gpuTime(n: 1, pos: pos + n) { e in self.encodeCommit(e) }))
        if pLmhTau != nil {
            r.append(("  lm_head FULL T=1", gpuTime(n: 1, pos: pos + n) { e in
                let saved = self.useSketchHead; self.useSketchHead = false; self.encodeLMHead(e, writeLogits: false); self.useSketchHead = saved }))
            r.append(("  sketch stage 1 T=1", gpuTime(n: 1, pos: pos + n) { e in
                e.setComputePipelineState(self.pLmhSketch[1]!)
                self.w(e, TensorID.embedHi, 0); self.w(e, TensorID.embedSB, 1); self.w(e, TensorID.finalNorm, 2)
                e.setBuffer(self.x, offset: 0, index: 3); e.setBuffer(self.lmhBounds, offset: 0, index: 4); e.setBuffer(self.state, offset: 0, index: 5)
                var k = self.sketchK; e.setBytes(&k, length: 4, index: 6)
                self.grid(e, Dims.vocab / 32, 1, 128) }))
            r.append(("  sketch tau", gpuTime(n: 1, pos: pos + n) { e in
                e.setComputePipelineState(self.pLmhTau!)
                e.setBuffer(self.lmhBounds, offset: 0, index: 0); e.setBuffer(self.lmhTau, offset: 0, index: 1); e.setBuffer(self.state, offset: 0, index: 2)
                var t32: Int32 = 1; e.setBytes(&t32, length: 4, index: 3)
                self.grid(e, 1, 1, 1024) }))
            r.append(("  sketch stage 2 T=1", gpuTime(n: 1, pos: pos + n) { e in
                e.setComputePipelineState(self.pLmhExact[1]!)
                self.w(e, TensorID.embed, 0); self.w(e, TensorID.embedSB, 1); self.w(e, TensorID.finalNorm, 2)
                e.setBuffer(self.x, offset: 0, index: 3); e.setBuffer(self.pval, offset: 0, index: 4); e.setBuffer(self.pidx, offset: 0, index: 5)
                e.setBuffer(self.state, offset: 0, index: 6); e.setBuffer(self.lmhBounds, offset: 0, index: 7); e.setBuffer(self.lmhTau, offset: 0, index: 8)
                self.grid(e, Dims.vocab / 32, 1, 128) }))
        }
        if pCommitSpec != nil {
            let s0 = state.contents().assumingMemoryBound(to: Int32.self)
            r.append(("  commit_spec T=2 (decode)", gpuTime(n: 1, pos: pos + n) { e in s0[8] = 2; s0[9] = 2; self.encodeCommitSpec(e) }))
            r.append(("  commit_spec T=3 (decode)", gpuTime(n: 1, pos: pos + n) { e in s0[8] = 3; s0[9] = 3; self.encodeCommitSpec(e) }))
        }
        return r
    }

    // MARK: generation

    public struct Timing {
        public var prefillTokens = 0
        public var tStart: UInt64 = 0
        public var tFirst: UInt64 = 0
        public var tEnd: UInt64 = 0
        public var stopToken: Int32 = -1
        public var tCommit: UInt64 = 0
        public var prefillCB: MTLCommandBuffer? = nil   // GPU start/end times for diagnostics
        public var passes = 0
        public var accepted = 0
    }

    public private(set) var lastTiming = Timing()

    /// Loads an n-gram draft table (NGramTable layout) for speculative decoding.
    public func loadDraftTable(path: String) throws {
        guard let d = FileManager.default.contents(atPath: path), d.count == NGramTable.slots * 16 else {
            throw EngineError("bad draft table \(path)")
        }
        ngram = d.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
        residency?.addAllocation(ngram)
        residency?.commit()
    }

    /// Submits a trivial dispatch so the GPU is awake (and clocked up) by the time real work arrives.
    /// Experiment: a background thread keeps a 32-thread spin kernel running on a second queue,
    /// so the GPU never looks idle to the clock governor (costs power).
    private var heaterOn = false
    public func startHeater(chunkIters: Int32 = 200_000) {
        guard !heaterOn, let p = pSpin else { return }
        heaterOn = true
        let q2 = queue.device.makeCommandQueue()!
        let sink = queue.device.makeBuffer(length: 16, options: .storageModeShared)!
        Thread.detachNewThread { [weak self] in
            var iters = chunkIters
            while self?.heaterOn == true {
                let cb = q2.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(p); e.setBuffer(sink, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1)
                e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
        }
    }
    public func stopHeater() { heaterOn = false }

    public func wake() {
        let (cb, e) = newEncoder()
        e.setComputePipelineState(pNoop)
        e.setBuffer(pval, offset: 0, index: 0)
        grid(e, 1, 1, 32)
        e.endEncoding()
        cb.commit()
    }

    // MARK: prefix KV cache on disk (a fresh process skips prefilling the system prompt)

    private func prefixKey() -> [UInt64] {
        var sb = stat()
        stat(model.path, &sb)
        return [0x3156_4B4C_4C49, UInt64(sb.st_size), UInt64(sb.st_mtimespec.tv_sec), UInt64(sb.st_mtimespec.tv_nsec), UInt64(maxContext)]
    }

    /// Saves keys/values of the first `tokens.count` cached tokens (they must be cached).
    public func savePrefix(path: String, tokens: [Int32]) {
        waitIdle()
        guard tokens.count <= cachedTokens.count, Array(cachedTokens.prefix(tokens.count)) == tokens else { return }
        var d = Data()
        for v in prefixKey() { withUnsafeBytes(of: v) { d.append(contentsOf: $0) } }
        withUnsafeBytes(of: UInt64(tokens.count)) { d.append(contentsOf: $0) }
        tokens.withUnsafeBytes { d.append(contentsOf: $0) }
        let row = Dims.headDim * 2, n = tokens.count
        for buf in [kc, vc] {
            for l in 0..<Dims.layers { for h in 0..<Dims.kvHeads {
                let off = ((l * Dims.kvHeads + h) * maxContext) * row
                d.append(Data(bytesNoCopy: buf.contents() + off, count: n * row, deallocator: .none))
            } }
        }
        try? FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        try? d.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// Loads a saved prefix if it was made for this model file; returns the prefix tokens.
    @discardableResult
    public func loadPrefix(path: String) -> [Int32]? {
        guard let d = FileManager.default.contents(atPath: path), d.count > 48 else { return nil }
        return d.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Int32]? in
            let key = prefixKey()
            for (i, v) in key.enumerated() where raw.loadUnaligned(fromByteOffset: 8 * i, as: UInt64.self) != v { return nil }
            let n = Int(raw.loadUnaligned(fromByteOffset: 40, as: UInt64.self))
            let row = Dims.headDim * 2
            guard n > 0, n < maxContext, d.count == 48 + 4 * n + 2 * Dims.layers * Dims.kvHeads * n * row else { return nil }
            let toks = (0..<n).map { raw.loadUnaligned(fromByteOffset: 48 + 4 * $0, as: Int32.self) }
            var src = 48 + 4 * n
            waitIdle()
            for buf in [kc, vc] {
                for l in 0..<Dims.layers { for h in 0..<Dims.kvHeads {
                    let off = ((l * Dims.kvHeads + h) * maxContext) * row
                    (buf.contents() + off).copyMemory(from: raw.baseAddress! + src, byteCount: n * row)
                    src += n * row
                } }
            }
            cachedTokens = toks
            return toks
        }
    }

    /// Candidate groups of the last sketch-head pass (row 0): groups whose upper bound reached tau.
    public func lastSketchCandidates() -> Int {
        waitIdle()
        let b = lmhBounds.contents().assumingMemoryBound(to: Float.self)
        let tau = lmhTau.contents().assumingMemoryBound(to: Float.self)[0]
        var c = 0
        for i in 0..<(Dims.vocab / 32) where b[Dims.vocab / 32 + i] >= tau { c += 1 }
        return c
    }

    /// Forget cached keys/values beyond the first `n` tokens (n = 0 clears the cache).
    public func truncateCache(_ n: Int) {
        if cachedTokens.count > n { cachedTokens.removeLast(cachedTokens.count - n) }
    }

    /// Blocks until all submitted GPU work has finished.
    public func waitIdle() {
        for cb in inflight { cb.waitUntilCompleted() }
        inflight.removeAll(keepingCapacity: true)
    }

    /// Greedy generation. `onToken` gets each token as soon as the GPU produces it (return false to stop).
    /// The stop token is not returned. `logitsOut`, if given, receives the logits of the first token.
    /// Decode steps per command buffer and how many steps the CPU encodes ahead.
    public var chunk = 1
    /// Reference runs: verify passes with no drafts (same kernels, so the output must not change).
    public var noDraft = ProcessInfo.processInfo.environment["ILL_NODRAFT"] != nil
    public var depth = Int(ProcessInfo.processInfo.environment["ILL_DEPTH"] ?? "") ?? 2
    public var firstChunk = 0

    @discardableResult
    public func generate(_ tokens: [Int32], maxNew: Int,
                         logitsOut: UnsafeMutablePointer<Float>? = nil,
                         onToken: ((Int32) -> Bool)? = nil) -> [Int32] {
        var timing = Timing()
        timing.tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        waitIdle()
        precondition(!tokens.isEmpty && maxNew >= 1 && maxNew <= maxSteps)
        var common = 0
        let lim = min(cachedTokens.count, tokens.count)
        while common < lim && cachedTokens[common] == tokens[common] { common += 1 }
        if common == tokens.count { common -= 1 }
        let n = tokens.count - common
        precondition(n <= maxPass && tokens.count + maxNew <= maxContext, "prompt too long")
        timing.prefillTokens = n

        let pt = promptToks.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<n { pt[i] = tokens[common + i] }
        let pr = promptRecent.contents().assumingMemoryBound(to: Int32.self)
        let np = min(tokens.count, 256)
        pr[0] = Int32(np)
        for i in 0..<np { pr[4 + i] = tokens[tokens.count - np + i] }
        let ot = outToks.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<maxNew { ot[i] = -1 }
        let s = state.contents().assumingMemoryBound(to: Int32.self)
        s[0] = Int32(common); s[1] = Int32(n); s[2] = 0; s[3] = 0
        s[4] = Int32(maxContext); s[5] = Int32(maxNew); s[6] = eos.0; s[7] = eos.1
        specActive = specT > 1 && (wideVerify ? (specT <= 16 && pG3Lmh[8] != nil) : pSpec[specT] != nil) && ngram.length == NGramTable.slots * 16
        s[8] = 1                                   // the prefill's lm_head produces one row
        s[9] = Int32(specActive ? specT : 1)
        s[10] = 0; s[11] = 0                       // passes, accepted drafts (diagnostics)
        s[13] = noDraft ? 1 : 0                  // pad2: drafting off (reference runs)

        // Prefill gets its own command buffer, committed at once. Decode steps go in small command
        // buffers kept `depth` steps ahead of the CPU: a token written by the GPU only becomes
        // CPU-visible once later memory traffic or the end of its command buffer flushes it, so
        // small buffers make every token (in particular the last one) visible within ~15 us.
        var (cb, e) = newEncoder()
        encodePrefill(e, n: n, writeLogits: logitsOut != nil)
        var encoded = 1
        while !specActive && encoded < min(maxNew, 1 + firstChunk) { encodeDecodeStep(e, writeLogits: false); encoded += 1 }
        e.endEncoding()
        cb.commit()
        timing.tCommit = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        timing.prefillCB = cb
        inflight.append(cb)

        var out: [Int32] = []
        out.reserveCapacity(maxNew)
        var stopped = false
        var got = 0
        while !stopped && got < maxNew {
            // Command buffers the GPU has finished: the prefill plus committed passes (speculative)
            // or one per token (plain). Staying only `depth` ahead keeps the GPU fed (a pass takes
            // ~3 ms, encoding one ~0.1 ms) without a long tail of early-exit passes after the stop,
            // which the next request would otherwise wait for.
            let done = specActive ? 1 + Int(ill_volatile_load(s + 10)) : got
            if encoded < maxNew && encoded - done <= depth {
                (cb, e) = newEncoder()
                let end = min(maxNew, encoded + chunk)
                // a speculative pass yields 1...specT tokens; a plain step exactly one
                while encoded < end {
                    if specActive {
                        if wideVerify { encodeWidePass(e, T: specT) } else { encodeSpecPass(e, T: specT) }
                    } else { encodeDecodeStep(e, writeLogits: false) }
                    encoded += 1
                }
                e.endEncoding()
                cb.commit()
                inflight.append(cb)
                continue
            }
            let t = ill_volatile_load(ot + got)
            if t < 0 { continue }
            if got == 0 {
                timing.tFirst = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                if let lo = logitsOut { lo.update(from: logits.contents().assumingMemoryBound(to: Float.self), count: Dims.vocab) }
            }
            got += 1
            if t == eos.0 || t == eos.1 { timing.stopToken = t; stopped = true; break }
            out.append(t)
            if let cbk = onToken, !cbk(t) { stopped = true }
        }
        timing.tEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        timing.passes = Int(ill_volatile_load(s + 10))
        timing.accepted = Int(ill_volatile_load(s + 11))
        lastTiming = timing
        // KV is valid for the prompt and every generated token except the last one.
        cachedTokens = tokens + out.dropLast()
        if stopped && got < encoded {
            // Remaining encoded steps early-exit on the done flag; if we stopped on our own (callback),
            // the GPU has not set it, so set it now to skip the rest.
            s[3] = 1
        }
        return out
    }
}
