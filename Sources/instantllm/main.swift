import Engine
import Foundation
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
                } else if ["logits-only", "quiet", "verbose", "stats"].contains(key) {
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

func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b &- a) / 1e6 }

switch argv[1] {
case "convert":
    guard let fmt = WeightFormat(name: args.get("format", "q4")) else { fail("bad --format") }
    do {
        try Converter.convert(hfDir: args.get("hf", "models/hf"), output: args.get("out", "model.ill"), format: fmt, quantFile: args.options["quant"])
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
                    print(String(format: "%2d prefill %3d ttft %6.2f e2e %7.2f gen %2d  %@", i, t.prefillTokens,
                                 ms(t.tStart, t.tFirst), ms(t.tStart, t.tEnd), out.count, out.map(String.init).joined(separator: ",")))
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
            for g in gold {
                forced.append(Int(engine.generate(ctx, maxNew: 1).first ?? engine.lastTiming.stopToken))
                ctx.append(Int32(g))
            }
        }
        let o: [String: Any] = ["output_ids": out.map { Int($0) }, "stop_token": stop >= 0 ? Int(stop) : NSNull(), "input_ids": ids,
                                "forced_argmax": forced]
        try! JSONSerialization.data(withJSONObject: o).write(to: URL(fileURLWithPath: String(format: "%@/p%02d.json", odir, i)))
        i += 1
    }
    print("wrote \(i) prompts to \(odir)")

case "profile":
    let engine: Engine
    do { engine = try Engine(modelPath: args.get("model", "model.ill")) } catch { fail("\(error)") }
    for (name, t) in engine.profile(n: args.int("n", 17), pos: args.int("pos", 46)) {
        print(String(format: "%-48@ %8.3f ms", name as NSString, t))
    }

default:
    fail("unknown command \(argv[1])")
}
