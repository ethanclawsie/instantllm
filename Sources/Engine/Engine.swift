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
    let pEmbedPrompt, pCommit, pAttnIn, pAttention, pOProj, pMlpIn, pMlpOut, pLMHead: MTLComputePipelineState
    let pRmsRows, pGemmQKV, pGemmO, pGemmGU, pGemmDown: MTLComputePipelineState
    let pRmsRowsH, pF32ToH: MTLComputePipelineState
    var pG2: [[MTLComputePipelineState]] = []  // [TF-1][qkv, o, gu, down]
    public var useGemm2 = true
    /// Prompts of at least this many new tokens use the GEMM prefill path.
    public var gemmThreshold = 2
    var inflight: [MTLCommandBuffer] = []

    /// Tokens whose keys/values are valid in the cache at positions 0..<count.
    public private(set) var cachedTokens: [Int32] = []

    public init(modelPath: String, maxContext: Int = 2048, maxPass: Int = 512, maxSteps: Int = 4096) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw EngineError("no Metal device") }
        device = dev
        model = try ModelFile(path: modelPath)
        format = model.format
        self.maxContext = maxContext
        self.maxPass = maxPass
        self.maxSteps = maxSteps
        queue = dev.makeCommandQueue()!

        guard let wb = dev.makeBuffer(bytesNoCopy: model.base, length: model.length,
                                      options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil)
        else { throw EngineError("cannot wrap model weights in a Metal buffer") }
        wbuf = wb

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
        state = buf(32)
        promptToks = buf(maxPass * 4)
        outToks = buf(maxSteps * 4)
        pval = buf(Dims.vocab / 32 * 4)
        pidx = buf(Dims.vocab / 32 * 4)
        logits = buf(Dims.vocab * 4)
        rope = buf(maxContext * 32 * 8)
        let r = rope.contents().assumingMemoryBound(to: Float.self)
        for p in 0..<maxContext {
            for i in 0..<32 {
                let invf = Float(1.0 / pow(Dims.ropeTheta, Double(2 * i) / 64.0))
                let a = Double(Float(p) * invf)
                r[(p * 32 + i) * 2] = Float(cos(a))
                r[(p * 32 + i) * 2 + 1] = Float(sin(a))
            }
        }

        let opts = MTLCompileOptions()
        opts.preprocessorMacros = ["FMT": NSNumber(value: model.format.rawValue)]
        opts.mathMode = .fast
        let lib = try dev.makeLibrary(source: shaderSource, options: opts)
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
        pRmsRows = try pipe("k_rmsnorm_rows")
        pGemmQKV = try pipe("k_gemm_qkv")
        pGemmO = try pipe("k_gemm_o")
        pGemmGU = try pipe("k_gemm_gu")
        pGemmDown = try pipe("k_gemm_down")
        pRmsRowsH = try pipe("k_rmsnorm_rows_h")
        pF32ToH = try pipe("k_f32_to_h")
        for tf in 1...4 {
            pG2.append(try ["qkv", "o", "gu", "down"].map { try pipe("k_g2_\($0)_\(tf)") })
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
        grid(e, 36, n)
        } else if part == 1 {
        e.setComputePipelineState(pAttention)
        e.setBuffer(q, offset: 0, index: 0)
        e.setBuffer(kc, offset: kvOff, index: 1)
        e.setBuffer(vc, offset: kvOff, index: 2)
        e.setBuffer(attn, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, Dims.heads, n)
        } else if part == 2 {
        e.setComputePipelineState(pOProj)
        w(e, TensorID.layer(l, TensorID.wo), 0)
        w(e, TensorID.layer(l, TensorID.woSB), 1)
        e.setBuffer(attn, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, Dims.hidden / 32, n)
        } else if part == 3 {
        e.setComputePipelineState(pMlpIn)
        w(e, TensorID.layer(l, TensorID.wgu), 0)
        w(e, TensorID.layer(l, TensorID.wguSB), 1)
        w(e, TensorID.layer(l, TensorID.mlpNorm), 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(hbuf, offset: 0, index: 4)
        e.setBuffer(state, offset: 0, index: 5)
        grid(e, Dims.ff / 16, n)
        } else {
        e.setComputePipelineState(pMlpOut)
        w(e, TensorID.layer(l, TensorID.wd), 0)
        w(e, TensorID.layer(l, TensorID.wdSB), 1)
        e.setBuffer(hbuf, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, Dims.hidden / 32, n)
        }
    }

    private func encodeGemm(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, a: MTLBuffer, wID: Int, sbID: Int,
                            biasID: Int?, out: MTLBuffer, kvOff: Int, rowsPerTG: Int, rows: Int, n: Int) {
        e.setComputePipelineState(p)
        e.setBuffer(a, offset: 0, index: 0)
        w(e, wID, 1)
        w(e, sbID, 2)
        w(e, biasID ?? sbID, 3)
        e.setBuffer(out, offset: 0, index: 4)
        e.setBuffer(kc, offset: kvOff, index: 5)
        e.setBuffer(vc, offset: kvOff, index: 6)
        e.setBuffer(rope, offset: 0, index: 7)
        e.setBuffer(state, offset: 0, index: 8)
        grid(e, rows / rowsPerTG, (n + 31) / 32, 128)
    }

    private func encodeRmsRows(_ e: MTLComputeCommandEncoder, normID: Int, n: Int) {
        e.setComputePipelineState(pRmsRows)
        e.setBuffer(x, offset: 0, index: 0)
        w(e, normID, 1)
        e.setBuffer(xn, offset: 0, index: 2)
        e.setBuffer(state, offset: 0, index: 3)
        grid(e, n, 1)
    }

    private func encodeG2(_ e: MTLComputeCommandEncoder, _ kind: Int, a: MTLBuffer, wID: Int, sbID: Int, biasID: Int?,
                          outf: MTLBuffer, outh: MTLBuffer, kvOff: Int, rows: Int, n: Int) {
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

    private func encodePrefillLayer2(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int) {
        let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
        let L = { TensorID.layer(l, $0) }
        e.setComputePipelineState(pRmsRowsH)
        e.setBuffer(x, offset: 0, index: 0); w(e, L(TensorID.attnNorm), 1); e.setBuffer(xnh, offset: 0, index: 2); e.setBuffer(state, offset: 0, index: 3)
        grid(e, n, 1)
        encodeG2(e, 0, a: xnh, wID: L(TensorID.wqkv), sbID: L(TensorID.wqkvSB), biasID: L(TensorID.bqkv), outf: q, outh: hh, kvOff: kvOff, rows: Dims.qkvRows, n: n)
        e.setComputePipelineState(pAttention)
        e.setBuffer(q, offset: 0, index: 0); e.setBuffer(kc, offset: kvOff, index: 1); e.setBuffer(vc, offset: kvOff, index: 2)
        e.setBuffer(attn, offset: 0, index: 3); e.setBuffer(state, offset: 0, index: 4)
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

    private func encodePrefillLayer(_ e: MTLComputeCommandEncoder, _ l: Int, n: Int) {
        let kvOff = l * Dims.kvHeads * maxContext * Dims.headDim * 2
        let L = { TensorID.layer(l, $0) }
        encodeRmsRows(e, normID: L(TensorID.attnNorm), n: n)
        encodeGemm(e, pGemmQKV, a: xn, wID: L(TensorID.wqkv), sbID: L(TensorID.wqkvSB), biasID: L(TensorID.bqkv),
                   out: q, kvOff: kvOff, rowsPerTG: 32, rows: Dims.qkvRows, n: n)
        e.setComputePipelineState(pAttention)
        e.setBuffer(q, offset: 0, index: 0)
        e.setBuffer(kc, offset: kvOff, index: 1)
        e.setBuffer(vc, offset: kvOff, index: 2)
        e.setBuffer(attn, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, Dims.heads, n)
        encodeGemm(e, pGemmO, a: attn, wID: L(TensorID.wo), sbID: L(TensorID.woSB), biasID: nil,
                   out: x, kvOff: kvOff, rowsPerTG: 32, rows: Dims.hidden, n: n)
        encodeRmsRows(e, normID: L(TensorID.mlpNorm), n: n)
        encodeGemm(e, pGemmGU, a: xn, wID: L(TensorID.wgu), sbID: L(TensorID.wguSB), biasID: nil,
                   out: hbuf, kvOff: kvOff, rowsPerTG: 64, rows: 2 * Dims.ff, n: n)
        encodeGemm(e, pGemmDown, a: hbuf, wID: L(TensorID.wd), sbID: L(TensorID.wdSB), biasID: nil,
                   out: x, kvOff: kvOff, rowsPerTG: 32, rows: Dims.hidden, n: n)
    }

    private func encodeLMHead(_ e: MTLComputeCommandEncoder, writeLogits: Bool) {
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
        grid(e, Dims.vocab / 32, 1)
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

    private func encodePrefill(_ e: MTLComputeCommandEncoder, n: Int, writeLogits: Bool) {
        e.setComputePipelineState(pEmbedPrompt)
        w(e, TensorID.embed, 0)
        w(e, TensorID.embedSB, 1)
        e.setBuffer(promptToks, offset: 0, index: 2)
        e.setBuffer(x, offset: 0, index: 3)
        e.setBuffer(state, offset: 0, index: 4)
        grid(e, n, 1)
        for l in 0..<Dims.layers {
            if n >= gemmThreshold { if useGemm2 { encodePrefillLayer2(e, l, n: n) } else { encodePrefillLayer(e, l, n: n) } } else { encodeLayer(e, l, n: n) }
        }
        encodeLMHead(e, writeLogits: writeLogits)
        encodeCommit(e)
    }

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
        r.append(("prefill pass (n=\(n)) incl lm_head+commit", gpuTime(n: n, pos: pos) { e in self.encodePrefill(e, n: n, writeLogits: false) }))
        let kv = { (l: Int) in l * Dims.kvHeads * self.maxContext * Dims.headDim * 2 }
        r.append(("  24x g2 qkv", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 0, a: self.xnh, wID: TensorID.layer(l, TensorID.wqkv), sbID: TensorID.layer(l, TensorID.wqkvSB), biasID: TensorID.layer(l, TensorID.bqkv), outf: self.q, outh: self.hh, kvOff: kv(l), rows: Dims.qkvRows, n: n) } }))
        r.append(("  24x attention", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers {
                e.setComputePipelineState(self.pAttention); e.setBuffer(self.q, offset: 0, index: 0); e.setBuffer(self.kc, offset: kv(l), index: 1)
                e.setBuffer(self.vc, offset: kv(l), index: 2); e.setBuffer(self.attn, offset: 0, index: 3); e.setBuffer(self.state, offset: 0, index: 4)
                self.grid(e, Dims.heads, n) } }))
        r.append(("  24x g2 o", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 1, a: self.attnh, wID: TensorID.layer(l, TensorID.wo), sbID: TensorID.layer(l, TensorID.woSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: Dims.hidden, n: n) } }))
        r.append(("  24x g2 gu", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 2, a: self.xnh, wID: TensorID.layer(l, TensorID.wgu), sbID: TensorID.layer(l, TensorID.wguSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: 2 * Dims.ff, n: n) } }))
        r.append(("  24x g2 down", gpuTime(n: n, pos: pos) { e in
            for l in 0..<Dims.layers { self.encodeG2(e, 3, a: self.hh, wID: TensorID.layer(l, TensorID.wd), sbID: TensorID.layer(l, TensorID.wdSB), biasID: nil, outf: self.x, outh: self.hh, kvOff: 0, rows: Dims.hidden, n: n) } }))
        r.append(("  lm_head + commit", gpuTime(n: n, pos: pos) { e in self.encodeLMHead(e, writeLogits: false); self.encodeCommit(e) }))
        r.append(("decode step (1 token)", gpuTime(n: 1, pos: pos + n) { e in self.encodeDecodeStep(e, writeLogits: false) }))
        r.append(("decode 16 steps / 16", gpuTime(n: 1, pos: pos + n, iters: 5) { e in for _ in 0..<16 { self.encodeDecodeStep(e, writeLogits: false) } } / 16))
        r.append(("  24x attn_in (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 0) } }))
        r.append(("  24x attention (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 1) } }))
        r.append(("  24x o_proj (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 2) } }))
        r.append(("  24x mlp_in (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 3) } }))
        r.append(("  24x mlp_out (decode)", gpuTime(n: 1, pos: pos + n) { e in for l in 0..<Dims.layers { self.encodeLayerPart(e, l, part: 4) } }))
        r.append(("  lm_head (decode)", gpuTime(n: 1, pos: pos + n) { e in self.encodeLMHead(e, writeLogits: false) }))
        r.append(("  commit (decode)", gpuTime(n: 1, pos: pos + n) { e in self.encodeCommit(e) }))
        return r
    }

    // MARK: generation

    public struct Timing {
        public var prefillTokens = 0
        public var tStart: UInt64 = 0
        public var tFirst: UInt64 = 0
        public var tEnd: UInt64 = 0
        public var stopToken: Int32 = -1
    }

    public private(set) var lastTiming = Timing()

    /// Blocks until all submitted GPU work has finished.
    public func waitIdle() {
        for cb in inflight { cb.waitUntilCompleted() }
        inflight.removeAll(keepingCapacity: true)
    }

    /// Greedy generation. `onToken` gets each token as soon as the GPU produces it (return false to stop).
    /// The stop token is not returned. `logitsOut`, if given, receives the logits of the first token.
    @discardableResult
    public func generate(_ tokens: [Int32], maxNew: Int, firstChunk: Int = 2, chunk: Int = 8,
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
        let ot = outToks.contents().assumingMemoryBound(to: Int32.self)
        for i in 0..<maxNew { ot[i] = -1 }
        let s = state.contents().assumingMemoryBound(to: Int32.self)
        s[0] = Int32(common); s[1] = Int32(n); s[2] = 0; s[3] = 0
        s[4] = Int32(maxContext); s[5] = Int32(maxNew); s[6] = eos.0; s[7] = eos.1

        var (cb, e) = newEncoder()
        encodePrefill(e, n: n, writeLogits: logitsOut != nil)
        var encoded = 1
        while encoded < min(maxNew, 1 + firstChunk) { encodeDecodeStep(e, writeLogits: false); encoded += 1 }
        e.endEncoding()
        cb.commit()
        inflight.append(cb)

        var out: [Int32] = []
        out.reserveCapacity(maxNew)
        var stopped = false
        var got = 0
        while !stopped && got < maxNew {
            if encoded < maxNew && encoded - got <= chunk {
                (cb, e) = newEncoder()
                let end = min(maxNew, encoded + chunk)
                while encoded < end { encodeDecodeStep(e, writeLogits: false); encoded += 1 }
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
