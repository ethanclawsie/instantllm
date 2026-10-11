import Engine
import Foundation
import Metal
import Tokenizer

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

struct Args {
    var positional: [String] = []
    var options: [String: String] = [:]
    var flags: Set<String> = []

    init(_ argv: ArraySlice<String>) {
        var it = argv.makeIterator()
        while let a = it.next() {
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if let eq = key.firstIndex(of: "=") {
                    options[String(key[..<eq])] = String(key[key.index(after: eq)...])
                } else if ["logits-only", "quiet", "verbose", "stats", "detach", "mmap", "ignore-eos", "no-prefix-cache", "sep", "sketch", "ttft-breakdown", "no-heater", "wide", "no-ildraft"].contains(key) {
                    flags.insert(key)
                } else if let v = it.next() {
                    options[key] = v
                }
            } else {
                positional.append(a)
            }
        }
    }

    func get(_ k: String, _ d: String) -> String { options[k] ?? d }
    func int(_ k: String, _ d: Int) -> Int { options[k].flatMap(Int.init) ?? d }
}

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    fail("""
    usage:
      instantllm convert --hf models/hf --out model.ill --format q4|f16
      instantllm ids --model model.ill --ids 1,2,3 [--max 64] [--logits out.bin]
    """)
}
let args = Args(argv.dropFirst(2))
let tMain = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b &- a) / 1e6 }

switch argv[1] {
case "convert":
    guard let fmt = WeightFormat(name: args.get("format", "q4")) else { fail("bad --format") }
    do {
        try Converter.convert(hfDir: args.get("hf", "models/hf"), output: args.get("out", "model.ill"), format: fmt, quantFile: args.options["quant"], sketch: args.flags.contains("sketch"))
    } catch { fail("convert failed: \(error)") }

case "ids":
    let ids = args.get("ids", "").split(separator: ",").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
    guard !ids.isEmpty else { fail("--ids required") }
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "model.ill")) } catch { fail("\(error)") }
    let reps = args.int("reps", 1)
    var logits = [Float](repeating: 0, count: Dims.vocab)
    for r in 0..<reps {
        let out = logits.withUnsafeMutableBufferPointer { lp in
            engine.generate(ids, maxNew: args.int("max", 64), logitsOut: args.options["logits"] != nil ? lp.baseAddress : nil)
        }
        let t = engine.lastTiming
        let dec = out.count > 1 ? Double(out.count - 1) / (ms(t.tFirst, t.tEnd) / 1e3) : 0
        if r == 0 || r == reps - 1 {
            print(out.map(String.init).joined(separator: ","))
        }
        print(String(format: "rep %d: prefill %d tok, ttft %.2f ms, total %.2f ms, %d tok, decode %.1f tok/s",
                     r, t.prefillTokens, ms(t.tStart, t.tFirst), ms(t.tStart, t.tEnd), out.count, dec))
        if args.flags.contains("stats") == false, args.options["fresh"] != nil { }
    }
    if let path = args.options["logits"] {
        let d = logits.withUnsafeBufferPointer { Data(buffer: $0) }
        try! d.write(to: URL(fileURLWithPath: path))
    }

case "idsbench":
    // Warm benchmark on pre-tokenized prompts (one comma-separated line per prompt): the shared
    // system prefix stays cached, each request prefills only its own suffix.
    let lines = (try? String(contentsOfFile: args.get("file", "scratch/prompts.ids"), encoding: .utf8)) ?? ""
    let prompts = lines.split(separator: "\n").map { $0.split(separator: ",").compactMap { Int32($0) } }
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "model.ill")) } catch { fail("\(error)") }
    let reps = args.int("reps", 3)
    let maxNew = args.int("max", 64)
    _ = engine.generate(prompts[0], maxNew: 4)
    var e2e: [Double] = [], ttft: [Double] = [], dec: [Double] = []
    for (i, p) in prompts.enumerated() {
        for r in 0..<reps {
            _ = engine.generate(prompts[(i + 1) % prompts.count], maxNew: 1) // evict this prompt's suffix
            let out = engine.generate(p, maxNew: maxNew)
            let t = engine.lastTiming
            if r == reps - 1 {
                e2e.append(ms(t.tStart, t.tEnd)); ttft.append(ms(t.tStart, t.tFirst))
                if out.count > 1 { dec.append(Double(out.count) / (ms(t.tFirst, t.tEnd) / 1e3)) }
                if args.flags.contains("verbose") {
                    print(String(format: "%2d prefill %3d commit %5.2f ttft %6.2f e2e %7.2f gen %2d  %@", i, t.prefillTokens,
                                 ms(t.tStart, t.tCommit), ms(t.tStart, t.tFirst), ms(t.tStart, t.tEnd), out.count, out.map(String.init).joined(separator: ",")))
                }
            }
        }
    }
    func med(_ a: [Double]) -> Double { let s = a.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
    print(String(format: "median e2e %.2f ms, median ttft %.2f ms, median decode %.1f tok/s (n=%d)", med(e2e), med(ttft), med(dec), e2e.count))

case "golden":
    // Runs every golden prompt and writes outputs in tools/reference's layout for compare.py.
    let gdir = args.get("golden", "tools/reference/golden")
    let odir = args.get("out", "scratch/engine_out")
    try? FileManager.default.createDirectory(atPath: odir, withIntermediateDirectories: true)
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "model.ill")) } catch { fail("\(error)") }
    if args.flags.contains("wide") {
        // Greedy decoding with the wide verify kernels (no drafts): the numerics of speculative runs.
        engine.wideVerify = true; engine.noDraft = true; engine.specT = 8
        do { try engine.loadDraftTable(path: args.get("draft-table", "models/ill/ngram.bin")) } catch { fail("\(error)") }
    }
    var logits = [Float](repeating: 0, count: Dims.vocab)
    var i = 0
    while let d = FileManager.default.contents(atPath: String(format: "%@/p%02d.json", gdir, i)),
          let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any], let ids = j["input_ids"] as? [Int] {
        let toks = ids.map { Int32($0) }
        let out = logits.withUnsafeMutableBufferPointer { engine.generate(toks, maxNew: 64, logitsOut: $0.baseAddress) }
        let stop = engine.lastTiming.stopToken
        try! logits.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: String(format: "%@/p%02d.logits.bin", odir, i)))
        // Teacher-forced argmax along the golden output (prefix cache makes each step one token).
        var forced: [Int] = []
        if let gold = j["output_ids"] as? [Int] {
            var ctx = toks
            for g in gold + [-1] {   // predictions for every golden token and the final stop token
                forced.append(Int(engine.generate(ctx, maxNew: 1).first ?? engine.lastTiming.stopToken))
                if g >= 0 { ctx.append(Int32(g)) }
            }
        }
        let o: [String: Any] = ["output_ids": out.map { Int($0) }, "stop_token": stop >= 0 ? Int(stop) : NSNull(), "input_ids": ids,
                                "forced_argmax": forced]
        try! JSONSerialization.data(withJSONObject: o).write(to: URL(fileURLWithPath: String(format: "%@/p%02d.json", odir, i)))
        i += 1
    }
    print("wrote \(i) prompts to \(odir)")

case "bench":
    // The competitor protocol (docs/SPEC.md, bench/competitors/*): prompt string -> final text.
    // TSV rows on stdout in the shared schema; a summary on stderr.
    let modelPath = args.get("model", "models/ill/qwen-q4.ill")
    let mode = args.get("mode", "prefix")
    let reps = args.int("reps", 3)
    let maxNew = args.int("n-predict", 64)
    let system = (try? String(contentsOfFile: args.get("system", "bench/system.txt"), encoding: .utf8)) ?? ""
    let prompts = ((try? String(contentsOfFile: args.get("prompts", "bench/prompts.txt"), encoding: .utf8)) ?? "")
        .split(separator: "\n").map(String.init)
    let tok: Tokenizer
    let engine: Engine
    do {
        tok = try Tokenizer(path: args.get("tokenizer", "models/ill/qwen.tkn"))
        engine = try Engine(modelPath: modelPath)
    } catch { fail("\(error)") }
    let variant = args.get("variant", engine.format.description)
    if args.flags.contains("ignore-eos") { engine.eos = (-5, -5) }
    engine.chunk = args.int("chunk", engine.chunk)
    engine.specT = args.int("spec", engine.defaultSpecT)
    if engine.specT > 1 {
        if (try? engine.loadDraftTable(path: args.get("draft-table", resolveAsset(nil, env: "INSTANTLLM_DRAFT", name: "ngram.bin")))) == nil { engine.specT = 1 }
    }
    engine.depth = args.int("depth", engine.depth)
    engine.firstChunk = args.int("first-chunk", engine.firstChunk)
    if !args.flags.contains("no-ildraft") {
        let p = args.options["ildraft"] ?? resolveAsset(nil, env: "INSTANTLLM_ILDRAFT", name: "ildraft.bin")
        if args.options["ildraft"] != nil || FileManager.default.fileExists(atPath: p) {
            do { try engine.loadLearnedDrafter(path: p) } catch { fail("\(error)") }
        }
    }
    engine.learnedT = args.int("learned-t", engine.learnedT)
    engine.learnedTree = args.int("learned-tree", engine.learnedTree)
    let prefixLen = tok.encodeWithSpecials("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n").count
    func esc(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\t", with: "\\t")
            .replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r")
    }
    func request(_ prompt: String) -> (text: String, t: Engine.Timing, gen: Int, t0: UInt64, t1: UInt64) {
        if mode == "full" { engine.truncateCache(0) } else { engine.truncateCache(prefixLen) }
        let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let ids = tok.encodeChat(system: system, user: prompt)
        if ids.count >= prefixLen + 5 { engine.setDraftContext(prompt: Array(ids[prefixLen..<(ids.count - 5)]), text: prompt, encode: tok.encode) }
        let out = engine.generate(ids, maxNew: maxNew)
        let text = tok.decode(out)
        let t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let t = engine.lastTiming
        return (text, t, out.count + (t.stopToken >= 0 ? 1 : 0), t0, t1)
    }
    for p in prompts { _ = request(p) }  // warmup (also caches the system prefix)
    if let ab = args.options["abba"] {
        // Paired comparison inside one process: per prompt run A, B, B, A (drift cancels).
        // --abba T compares speculation width T with plain decoding, --abba A,B two widths.
        // "w8" = a wide verify pass (Neural Accelerators) with 8 rows; an "l" prefix = learned drafter;
        // "lt15" = learned drafter building a 15-draft tree (wide passes of 16 rows).
        let parse = { (x0: Substring) -> (Int, Bool, Bool, Int) in
            if x0.hasPrefix("lt") { let b = Int(x0.dropFirst(2)) ?? 7; return (b + 1, true, true, b) }
            let l = x0.hasPrefix("l"), x = l ? x0.dropFirst() : x0
            return x.hasPrefix("w") ? (Int(x.dropFirst()) ?? 8, true, l, 0) : (Int(x) ?? 2, false, l, 0)
        }
        let w = ab.split(separator: ",").map(parse)
        let (aTree, bTree) = w.count == 2 ? (w[0].3, w[1].3) : (0, w.first?.3 ?? 0)
        let (aT, abT) = w.count == 2 ? (w[0].0, w[1].0) : (1, w.first?.0 ?? 2)
        let (aWide, bWide) = w.count == 2 ? (w[0].1, w[1].1) : (false, w.first?.1 ?? false)
        let (aL, bL) = w.count == 2 ? (w[0].2, w[1].2) : (false, w.first?.2 ?? false)
        try? engine.loadDraftTable(path: args.get("draft-table", "models/ill/ngram.bin"))
        var dE: [Double] = [], dT: [Double] = [], plainE: [Double] = [], specE: [Double] = []
        for _ in 0..<reps {
            for p in prompts {
                var e = [Double](repeating: 0, count: 4), t = [Double](repeating: 0, count: 4)
                for (i, T) in [aT, abT, abT, aT].enumerated() {
                    let (wide, learned, tr) = (i == 1 || i == 2) ? (bWide, bL, bTree) : (aWide, aL, aTree)
                    engine.useLearned = learned
                    engine.learnedTree = tr
                    if learned { engine.learnedT = T; engine.learnedWide = wide } else { engine.specT = T; engine.wideVerify = wide }
                    let res = request(p)
                    e[i] = ms(res.t0, res.t1); t[i] = ms(res.t0, res.t.tFirst)
                }
                dE.append((e[1] + e[2] - e[0] - e[3]) / 2); dT.append((t[1] + t[2] - t[0] - t[3]) / 2)
                plainE.append((e[0] + e[3]) / 2); specE.append((e[1] + e[2]) / 2)
            }
        }
        func med(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
        let mean = { (a: [Double]) in a.reduce(0, +) / Double(a.count) }
        print(String(format: "ABBA spec %@%d vs %@%d: e2e mean %.2f -> %.2f ms (%.1f%%), median paired delta %.2f ms; ttft median delta %.2f ms",
                     (bL ? "l" : "") + (bTree > 0 ? "t" : (bWide ? "w" : "")), abT, (aL ? "l" : "") + (aTree > 0 ? "t" : (aWide ? "w" : "")), aT, mean(plainE), mean(specE), 100 * (mean(specE) / mean(plainE) - 1), med(dE), med(dT)))
        exit(0)
    }
    print("engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext")
    var e2e: [Double] = [], ttft: [Double] = [], dec: [Double] = []
    var totPasses = 0, totAcc = 0, totGen = 0
    let gapUs = useconds_t(args.int("gap-ms", 0) * 1000)   // idle time before each request
    if args.options["heater-iters"] != nil { engine.startHeater(chunkIters: Int32(args.int("heater-iters", 200_000))) }
    for (i, p) in prompts.enumerated() {
        for r in 0..<reps {
            if gapUs > 0 { usleep(gapUs) }
            let res = request(p)
            totPasses += res.t.passes; totAcc += res.t.accepted; totGen += res.gen
            if args.flags.contains("verbose") { FileHandle.standardError.write("p\(i) candidates(last pass) \(engine.lastSketchCandidates())\n".data(using: .utf8)!) }
            if args.flags.contains("ttft-breakdown") {
                var gs = 0.0, gd = 0.0
                if let cb = res.t.prefillCB { cb.waitUntilCompleted(); gs = cb.gpuStartTime * 1e3 - Double(res.t.tCommit) / 1e6; gd = (cb.gpuEndTime - cb.gpuStartTime) * 1e3 }
                FileHandle.standardError.write(String(format: "p%d n=%d tokenize %.3f  encode+commit %.3f  commit->first %.3f ms  commit->gpu %.3f  gpu %.3f\n", i, res.t.prefillTokens,
                    ms(res.t0, res.t.tStart), ms(res.t.tStart, res.t.tCommit), ms(res.t.tCommit, res.t.tFirst), gs, gd).data(using: .utf8)!)
            }
            let ttftMs = ms(res.t0, res.t.tFirst)
            let decTs = res.gen > 1 ? Double(res.gen - 1) / (ms(res.t.tFirst, res.t.tEnd) / 1e3) : Double.nan
            e2e.append(ms(res.t0, res.t1)); ttft.append(ttftMs); if res.gen > 1 { dec.append(decTs) }
            print("instantllm\t\(variant)\t\(mode)\t\(i)\t\(r)\t" + String(format: "%.3f\t%.3f", ms(res.t0, res.t1), ttftMs)
                  + "\t\(res.t.prefillTokens)\t\(res.gen)\t" + String(format: "%.1f", decTs) + "\t" + esc(res.text))
        }
    }
    func med(_ a: [Double]) -> Double { let s = a.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }
    FileHandle.standardError.write(String(format: "instantllm %@ %@: median e2e %.2f ms, ttft %.2f ms, decode %.1f tok/s (passes %d, accepted drafts %d, tokens %d)\n",
                                          variant, mode, med(e2e), med(ttft), med(dec), totPasses, totAcc, totGen).data(using: .utf8)!)

case "run":
    // One-shot: tokenize, generate, stream the command to stdout.
    let system = args.options["system"] ?? ((try? String(contentsOfFile: args.get("system-file", "bench/system.txt"), encoding: .utf8)) ?? "")
    let prompt = args.positional.joined(separator: " ")
    let tok: Tokenizer
    let engine: Engine
    do {
        tok = try Tokenizer(path: args.get("tokenizer", "models/ill/qwen.tkn"))
        engine = try Engine(modelPath: args.get("model", "models/ill/qwen-q4.ill"), pinMemory: false)
    } catch { fail("\(error)") }
    if !args.flags.contains("no-heater") { engine.startHeater() }   // the GPU clock ramps while we load
    engine.specT = args.int("spec", engine.defaultSpecT)
    if engine.specT > 1 && (try? engine.loadDraftTable(path: args.get("draft-table", resolveAsset(nil, env: "INSTANTLLM_DRAFT", name: "ngram.bin")))) == nil { engine.specT = 1 }
    // System-prompt KV saved by an earlier run (keyed by model file and the exact prefix tokens).
    let prefixToks = tok.encodeWithSpecials("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n")
    var hasher = Hasher()
    hasher.combine(prefixToks)
    let cacheDir = FileManager.default.homeDirectoryForCurrentUser.path + "/Library/Caches/instantllm"
    let prefixFile = cacheDir + "/prefix-" + String(UInt64(bitPattern: Int64(prefixToks.reduce(Int64(1469598103934665603)) { ($0 ^ Int64($1)) &* 1099511628211 })), radix: 16) + ".kv"
    _ = hasher
    let hadPrefix = args.flags.contains("no-prefix-cache") ? false : (engine.loadPrefix(path: prefixFile) == prefixToks)
    if !args.flags.contains("no-ildraft") {
        try? engine.loadLearnedDrafter(path: args.get("ildraft", resolveAsset(nil, env: "INSTANTLLM_ILDRAFT", name: "ildraft.bin")))
    }
    let tInit = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let ids = tok.encodeChat(system: system, user: prompt)
    if ids.count >= prefixToks.count + 5 {
        engine.setDraftContext(prompt: Array(ids[prefixToks.count..<(ids.count - 5)]), text: prompt, encode: tok.encode)
    }
    let out = FileHandle.standardOutput
    defer {
        if args.flags.contains("verbose") {
            // process start time from the kernel
            var kp = kinfo_proc(); var sz = MemoryLayout<kinfo_proc>.size
            var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
            sysctl(&mib, 4, &kp, &sz, nil, 0)
            var tv = timeval(); gettimeofday(&tv, nil)
            let sinceStart = Double(tv.tv_sec - kp.kp_proc.p_starttime.tv_sec) * 1e3 + Double(tv.tv_usec - kp.kp_proc.p_starttime.tv_usec) / 1e3
            let t = engine.lastTiming
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            FileHandle.standardError.write(String(format: "process start -> now %.1f ms; main->init done %.1f ms; ttft %.1f ms; gen %.1f ms; prefill %d tokens\n",
                sinceStart, ms(tMain, tInit), ms(tInit, t.tFirst), ms(t.tFirst, t.tEnd), t.prefillTokens).data(using: .utf8)!)
            _ = now
        }
    }
    engine.generate(ids, maxNew: args.int("n-predict", 64)) { t in
        let b = tok.tokenBytes(t)
        if let base = b.baseAddress { out.write(Data(bytes: base, count: b.count)) }
        return true
    }
    out.write("\n".data(using: .utf8)!)
    if args.flags.contains("stats") {
        let t = engine.lastTiming
        let gen = (engine.cachedTokens.count - ids.count + 1) + (t.stopToken >= 0 ? 1 : 0)
        let dec = gen > 1 ? Double(gen - 1) / (ms(t.tFirst, t.tEnd) / 1e3) : Double.nan
        FileHandle.standardError.write("gen_tokens=\(gen) prefill=\(t.prefillTokens) decode_tok_s=\(String(format: "%.1f", dec))\n".data(using: .utf8)!)
    }
    if !hadPrefix && !args.flags.contains("no-prefix-cache") { engine.savePrefix(path: prefixFile, tokens: prefixToks) }

case "batch":
    // prompts file (one per line) -> JSONL {"prompt", "ids", "text"} with the model's greedy outputs.
    let system = (try? String(contentsOfFile: args.get("system", "bench/system.txt"), encoding: .utf8)) ?? ""
    let lines = ((try? String(contentsOfFile: args.get("prompts", "bench/prompts.txt"), encoding: .utf8)) ?? "")
        .split(separator: "\n").map(String.init).filter { !$0.isEmpty }
    let tok: Tokenizer
    let engine: Engine
    do {
        tok = try Tokenizer(path: args.get("tokenizer", "models/ill/qwen.tkn"))
        engine = try Engine(modelPath: args.get("model", "models/ill/qwen-q4.ill"))
    } catch { fail("\(error)") }
    let outPath = args.get("out", "scratch/batch.jsonl")
    FileManager.default.createFile(atPath: outPath, contents: nil)
    let fh = FileHandle(forWritingAtPath: outPath)!
    for p in lines {
        let ids = tok.encodeChat(system: system, user: p)
        let out = engine.generate(ids, maxNew: args.int("n-predict", 64))
        let rec: [String: Any] = ["prompt": p, "ids": out.map { Int($0) }, "text": tok.decode(out), "stop": Int(engine.lastTiming.stopToken)]
        fh.write(try! JSONSerialization.data(withJSONObject: rec))
        fh.write("\n".data(using: .utf8)!)
    }
    fh.closeFile()
    print("wrote \(lines.count) outputs to \(outPath)")

case "ngram":
    // Builds the speculative-decoding draft table from JSONL outputs ({"ids": [...], "stop": id}).
    var seqs: [[Int32]] = []
    for path in args.positional {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { fail("cannot read \(path)") }
        for line in text.split(separator: "\n") {
            guard let d = line.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let ids = j["ids"] as? [Int] else { continue }
            var seq = ids.map { Int32($0) }
            if let stop = j["stop"] as? Int, stop >= 0 { seq.append(Int32(stop)) }
            seqs.append(seq)
        }
    }
    let table = NGramTable.build(seqs)
    try! table.write(to: URL(fileURLWithPath: args.get("out", "models/ill/ngram.bin")))
    print("built draft table from \(seqs.count) sequences")

case "tokenize":
    do {
        if let json = args.options["compile"] {
            try Tokenizer.compile(tokenizerJSON: json, output: args.get("out", "models/ill/qwen.tkn"))
            print("compiled tokenizer")
        }
    } catch { fail("\(error)") }

case "idletest":
    // Latency of a request after the process sat idle for a while (GPU power / residency effects).
    let tok: Tokenizer
    let engine: Engine
    do {
        tok = try Tokenizer(path: args.get("tokenizer", "models/ill/qwen.tkn"))
        engine = try Engine(modelPath: args.get("model", "models/ill/qwen-q4.ill"))
    } catch { fail("\(error)") }
    let system = (try? String(contentsOfFile: "bench/system.txt", encoding: .utf8)) ?? ""
    let ids = tok.encodeChat(system: system, user: "kill all processes named node")
    _ = engine.generate(ids, maxNew: 8)
    engine.truncateCache(0)
    _ = engine.generate(ids, maxNew: 8)
    for idle in [0.0, 0.5, 1, 2, 3, 5, 10] {
        usleep(useconds_t(idle * 1e6))
        let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let out = engine.generate(ids, maxNew: 32)
        let t = engine.lastTiming
        print(String(format: "idle %4.1f s: ttft %7.2f ms  e2e %7.2f ms  (%d tokens)", idle, ms(t0, t.tFirst), ms(t0, t.tEnd), out.count))
    }

case "ramp":
    // GPU clock ramp after idle: per-token intervals of a plain (one step per command buffer)
    // generation that starts after `gap` seconds without GPU work.
    let tok: Tokenizer
    let engine: Engine
    do {
        tok = try Tokenizer(path: args.get("tokenizer", "models/ill/qwen.tkn"))
        engine = try Engine(modelPath: args.get("model", "models/ill/qwen-q4.ill"))
    } catch { fail("\(error)") }
    let system = (try? String(contentsOfFile: "bench/system.txt", encoding: .utf8)) ?? ""
    let ids = tok.encodeChat(system: system, user: "kill all processes named node")
    engine.eos = (-5, -5)
    _ = engine.generate(ids, maxNew: 48)
    for gap in [0.0, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0] {
        usleep(useconds_t(gap * 1e6))
        var ts: [UInt64] = []
        let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        _ = engine.generate(ids, maxNew: 40) { _ in ts.append(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)); return true }
        var line = String(format: "gap %4.2f s: ttft %5.2f | steps", gap, ms(t0, ts[0]))
        for i in 1..<min(ts.count, 40) where i <= 12 || i % 4 == 0 { line += String(format: " %.2f", ms(ts[i - 1], ts[i])) }
        print(line)
    }

case "heat":
    // Runs the GPU heater (see Engine.startHeater) for --seconds, for power measurements.
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "models/ill/qwen-q4.ill")) } catch { fail("\(error)") }
    engine.startHeater(chunkIters: Int32(args.int("heater-iters", 200_000)))
    usleep(useconds_t(args.int("seconds", 20) * 1_000_000))
    engine.stopHeater()

case "serve":
    serve(args: args)

case "profile":
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "model.ill")) } catch { fail("\(error)") }
    for (name, t) in engine.profile(n: args.int("n", 17), pos: args.int("pos", 46)) {
        print(String(format: "%-48@ %8.3f ms", name as NSString, t))
    }

default:
    fail("unknown command \(argv[1])")
}
