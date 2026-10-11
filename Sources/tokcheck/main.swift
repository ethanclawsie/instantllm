// tokcheck: verify the Swift tokenizer against HF golden data and report timings.
//
//   tokcheck [all]                        compile + test + bench (default paths below)
//   tokcheck compile [tokenizer.json] [out.tkn]
//   tokcheck test [file.tkn] [golden.bin]
//   tokcheck bench [file.tkn]
//   tokcheck coldstart [file.tkn]         one fresh-process measurement: init + first encodeChat
//   tokcheck encode [--specials] text...  print ids
//
// Run from the repo root. Golden data: tools/tokenizer_golden.py -> scratch/tok/golden.bin
import Darwin
import Foundation
import Tokenizer

let defaultJSON = "models/hf/tokenizer.json"
let defaultTKN = "scratch/tok/tokenizer.tkn"
let defaultGolden = "scratch/tok/golden.bin"

@inline(__always) func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

func fail(_ s: String) -> Never {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    exit(1)
}

func stats(_ xs: [Double]) -> (min: Double, med: Double) {
    let s = xs.sorted()
    return (s.first ?? .nan, s.isEmpty ? .nan : s[s.count / 2])
}

func f2(_ x: Double) -> String { String(format: "%.2f", x) }
func f1(_ x: Double) -> String { String(format: "%.1f", x) }

func escape(_ s: String, max: Int = 160) -> String {
    var out = ""
    for u in s.unicodeScalars {
        switch u {
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if u.value < 0x20 || (u.value >= 0x7F && u.value < 0xA0) || u.properties.generalCategory == .format
                || u.properties.generalCategory == .nonspacingMark || u.properties.isWhitespace {
                out += String(format: "\\u{%X}", u.value)
            } else {
                out.unicodeScalars.append(u)
            }
        }
        if out.count > max { return out + "…" }
    }
    return out
}

// MARK: golden file

struct Case {
    var mode: UInt8
    var name: String
    var a: String
    var b: String
    var ids: [Int32]
    var exp: String
}

func loadGolden(_ path: String) -> [Case] {
    guard let d = FileManager.default.contents(atPath: path) else {
        fail("cannot read \(path) (run: .venv/bin/python -I tools/tokenizer_golden.py)")
    }
    return d.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Case] in
        var pos = 0
        func u8() -> UInt8 { defer { pos += 1 }; return raw[pos] }
        func u32() -> Int { defer { pos += 4 }; return Int(raw.loadUnaligned(fromByteOffset: pos, as: UInt32.self)) }
        func str(_ n: Int) -> String {
            defer { pos += n }
            return String(decoding: UnsafeRawBufferPointer(rebasing: raw[pos..<(pos + n)]), as: UTF8.self)
        }
        guard raw.count >= 12, String(decoding: UnsafeRawBufferPointer(rebasing: raw[0..<8]), as: UTF8.self) == "TKGOLD01"
        else { fail("bad golden file") }
        pos = 8
        let n = u32()
        var cases = [Case]()
        cases.reserveCapacity(n)
        for _ in 0..<n {
            let mode = u8()
            let name = str(Int(u8()))
            let a = str(u32())
            let b = str(u32())
            let k = u32()
            var ids = [Int32](repeating: 0, count: k)
            ids.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[pos..<(pos + 4 * k)])) }
            pos += 4 * k
            let exp = str(u32())
            cases.append(Case(mode: mode, name: name, a: a, b: b, ids: ids, exp: exp))
        }
        return cases
    }
}

// MARK: test

func runTest(_ tok: Tokenizer, _ goldenPath: String) -> Bool {
    let t0 = nowNs()
    let cases = loadGolden(goldenPath)
    let t1 = nowNs()
    print("golden: \(cases.count) cases from \(goldenPath) (\(f1(Double(t1 - t0) / 1e6)) ms to load)")
    var perCat = [String: (total: Int, idFail: Int, decFail: Int, rtFail: Int, streamFail: Int)]()
    var order = [String]()
    var shown = 0
    var totalIds = 0
    let modeNames = ["plain", "specials", "chat", "decode", "nfc"]
    var perMode = [Int](repeating: 0, count: 5)
    var failedCases = 0
    var caseFailed = false
    func report(_ c: Case, _ got: [Int32], _ what: String) {
        caseFailed = true
        guard shown < 25 else { return }
        shown += 1
        print("  FAIL [\(c.name) \(modeNames[Int(c.mode)])] \(what)")
        print("    text: \"\(escape(c.a))\"" + (c.b.isEmpty ? "" : "  user: \"\(escape(c.b))\""))
        if c.mode <= 2 {
            var k = 0
            while k < min(got.count, c.ids.count) && got[k] == c.ids[k] { k += 1 }
            let lo = max(0, k - 3)
            func show(_ ids: [Int32]) -> String {
                ids[lo..<min(ids.count, k + 6)].map { "\($0)\"\(escape(tok.decode([$0]), max: 20))\"" }
                    .joined(separator: " ")
            }
            print("    first diff at token \(k): expected \(show(c.ids))")
            print("                         got      \(show(got))")
        }
    }
    // streaming detokenizer: concatenated chunks must equal decode(ids)
    func streamOK(_ ids: [Int32], _ exp: String) -> Bool {
        var st = TextStream(tok)
        var out = ""
        for id in ids { out += st.push(id) }
        out += st.finish()
        return out == exp
    }
    let tStart = nowNs()
    for (ci, c) in cases.enumerated() {
        if perCat[c.name] == nil {
            perCat[c.name] = (0, 0, 0, 0, 0)
            order.append(c.name)
        }
        var e = perCat[c.name]!
        e.total += 1
        perMode[Int(c.mode)] += 1
        caseFailed = false
        switch c.mode {
        case 0, 1, 2:
            let got: [Int32]
            if c.mode == 0 {
                got = tok.encode(c.a)
            } else if c.mode == 1 {
                got = tok.encodeWithSpecials(c.a)
            } else {
                got = tok.encodeChat(system: c.a, user: c.b)
            }
            totalIds += got.count
            if got != c.ids {
                e.idFail += 1
                report(c, got, "token ids differ (expected \(c.ids.count), got \(got.count))")
            }
            let dec = tok.decode(got)
            if dec != c.exp {
                e.decFail += 1
                report(c, got, "decode differs: \"\(escape(dec))\" vs \"\(escape(c.exp))\"")
            }
            if c.mode < 2 && dec != tok.normalize(c.a) {
                e.rtFail += 1
                report(c, got, "round trip decode(encode(x)) != NFC(x)")
            }
            if ci % 7 == 0 && !streamOK(got, c.exp) {
                e.streamFail += 1
                report(c, got, "TextStream output differs from decode")
            }
        case 3:
            let dec = tok.decode(c.ids)
            if dec != c.exp {
                e.decFail += 1
                report(c, c.ids, "decode of \(c.ids) = \"\(escape(dec))\" vs \"\(escape(c.exp))\"")
            }
            if !streamOK(c.ids, c.exp) {
                e.streamFail += 1
                report(c, c.ids, "TextStream output differs from decode of \(c.ids)")
            }
        default:
            let n = tok.normalize(c.a)
            if n != c.exp {
                e.decFail += 1
                report(c, [], "NFC: \"\(escape(n))\" vs \"\(escape(c.exp))\"")
            }
        }
        if caseFailed { failedCases += 1 }
        perCat[c.name] = e
    }
    let tEnd = nowNs()

    // bridged (non-native) strings and concurrent use must give identical results
    let sample = cases.enumerated().filter { $0.element.mode <= 1 && $0.offset % 40 == 0 }.map { $0.element }
    var bridgedFails = 0
    for c in sample {
        // (NSString drops a leading U+FEFF, so compare with a native copy when the content changed)
        let b = NSString(string: c.a) as String
        let got = c.mode == 0 ? tok.encode(b) : tok.encodeWithSpecials(b)
        let native = String(decoding: Array(b.utf8), as: UTF8.self)
        let want = b.utf8.elementsEqual(c.a.utf8) ? c.ids
            : (c.mode == 0 ? tok.encode(native) : tok.encodeWithSpecials(native))
        if got != want { bridgedFails += 1 }
    }
    let lock = NSLock()
    var concFails = 0
    DispatchQueue.concurrentPerform(iterations: 8) { t in
        var bad = 0
        for _ in 0..<3 {
            for (k, c) in sample.enumerated() where k % 2 == t % 2 {
                let got = c.mode == 0 ? tok.encode(c.a) : tok.encodeWithSpecials(c.a)
                if got != c.ids || tok.decode(got) != c.exp { bad += 1 }
            }
        }
        lock.lock()
        concFails += bad
        lock.unlock()
    }
    var lines = [String]()
    for k in order {
        let e = perCat[k]!
        if e.idFail + e.decFail + e.rtFail + e.streamFail > 0 {
            lines.append("  \(k): \(e.total) cases, ids \(e.idFail), decode \(e.decFail), round-trip \(e.rtFail), stream \(e.streamFail)")
        }
    }
    print("modes: " + (0..<5).map { "\(modeNames[$0])=\(perMode[$0])" }.joined(separator: " "))
    print("categories: \(order.count); checked \(cases.count) cases, \(totalIds) tokens in \(f1(Double(tEnd - tStart) / 1e6)) ms")
    print("extra: bridged NSString \(sample.count - bridgedFails)/\(sample.count) ok, 8-thread concurrent mismatches \(concFails)")
    let allOK = failedCases == 0 && bridgedFails == 0 && concFails == 0
    if allOK {
        print("RESULT: 100% exact match (\(cases.count)/\(cases.count) cases: token ids, decode, round trip, NFC, TextStream)")
    } else {
        print("RESULT: \(failedCases) of \(cases.count) cases FAILED (+\(bridgedFails) bridged, +\(concFails) concurrent):")
        lines.forEach { print($0) }
    }
    return allOK
}

/// Truncated / corrupted / foreign files must be rejected by init(path:) with an error, not crash.
func runRobustness(_ path: String) -> Bool {
    guard let good = FileManager.default.contents(atPath: path) else { return false }
    let tmp = NSTemporaryDirectory() + "tokcheck-bad-\(getpid()).tkn"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    var bad: [Data] = [Data(), good.prefix(100), good.prefix(good.count / 2), good.prefix(good.count - 1),
                       FileManager.default.contents(atPath: defaultJSON)?.prefix(1 << 20) ?? Data()]
    for field in [3, 4, 10, 11, 12, 13, 14, 20, 21, 25, 28, 29] {  // header words: counts, bits, offsets
        var d = good
        d.withUnsafeMutableBytes { $0.storeBytes(of: UInt32(0x7FFF_FFF0), toByteOffset: 4 * field, as: UInt32.self) }
        bad.append(d)
    }
    var rejected = 0
    for d in bad {
        FileManager.default.createFile(atPath: tmp, contents: d)
        if (try? Tokenizer(path: tmp)) == nil { rejected += 1 }
    }
    print("robustness: \(rejected)/\(bad.count) truncated/corrupt files rejected by init(path:)")
    return rejected == bad.count
}

// MARK: bench

func readLines(_ path: String) -> [String] {
    guard let s = try? String(contentsOfFile: path, encoding: .utf8) else { fail("cannot read \(path)") }
    return s.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
}

@inline(never) func blackHole<T>(_ x: T) { withExtendedLifetime(x) {} }

func runBench(_ path: String, _ goldenPath: String?) {
    let system = (try? String(contentsOfFile: "bench/system.txt", encoding: .utf8)) ?? "You are a helpful assistant."
    let prompts = readLines("bench/prompts.txt")
    print("--- load (\(path), \((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) bytes)")
    var tInit = [Double](), tFirst = [Double](), tFirstPF = [Double](), tPF = [Double]()
    for i in 0..<400 {
        let a = nowNs()
        let t = try! Tokenizer(path: path)
        let b = nowNs()
        if i % 2 == 0 {
            let ids = t.encodeChat(system: system, user: prompts[i % prompts.count])
            let c = nowNs()
            blackHole(ids)
            tFirst.append(Double(c - b) / 1e3)
        } else {
            t.prefetch()
            let c = nowNs()
            let ids = t.encodeChat(system: system, user: prompts[i % prompts.count])
            let d = nowNs()
            blackHole(ids)
            tPF.append(Double(c - b) / 1e3)
            tFirstPF.append(Double(d - c) / 1e3)
        }
        tInit.append(Double(b - a) / 1e3)
    }
    let si = stats(tInit), sf = stats(tFirst), sp = stats(tPF), sfp = stats(tFirstPF)
    print("init(path:)                         min \(f2(si.min)) us   median \(f2(si.med)) us")
    print("first encodeChat after init         min \(f2(sf.min)) us   median \(f2(sf.med)) us  (fresh mapping: page faults)")
    print("prefetch() after init               min \(f2(sp.min)) us   median \(f2(sp.med)) us")
    print("first encodeChat after prefetch     min \(f2(sfp.min)) us   median \(f2(sfp.med)) us")

    let tok = try! Tokenizer(path: path)
    print("--- warm encode (20 bench prompts, 3000 iterations each; per-call times)")
    func bench(_ label: String, _ f: (Int) -> [Int32]) {
        var mins = [Double](), meds = [Double]()
        var ntok = 0
        for (pi, _) in prompts.enumerated() {
            var xs = [Double]()
            xs.reserveCapacity(3000)
            for _ in 0..<200 { blackHole(f(pi)) }
            for _ in 0..<3000 {
                let a = nowNs()
                let r = f(pi)
                let b = nowNs()
                blackHole(r)
                xs.append(Double(b - a) / 1e3)
            }
            let s = stats(xs)
            mins.append(s.min)
            meds.append(s.med)
            ntok += f(pi).count
        }
        let avgMed = meds.reduce(0, +) / Double(meds.count)
        print("\(label.padding(toLength: 36, withPad: " ", startingAt: 0))min \(f2(mins.min()!)) us   median \(f2(stats(meds).med)) us   mean-of-medians \(f2(avgMed)) us   (\(f1(Double(ntok) / Double(prompts.count))) tokens avg)")
    }
    let rendered = prompts.map {
        "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\($0)<|im_end|>\n<|im_start|>assistant\n"
    }
    bench("encodeChat(system:user:)") { tok.encodeChat(system: system, user: prompts[$0]) }
    bench("encode(user prompt)") { tok.encode(prompts[$0]) }
    bench("encodeWithSpecials(rendered)") { tok.encodeWithSpecials(rendered[$0]) }
    let outs = prompts.map { tok.encodeChat(system: system, user: $0) }
    bench("decode(chat ids)") { blackHole(tok.decode(outs[$0])); return [] }

    if let gp = goldenPath, FileManager.default.fileExists(atPath: gp) {
        print("--- throughput (golden corpus, plain encode)")
        let cases = loadGolden(gp).filter { $0.mode == 0 }
        func group(_ label: String, _ pred: (String) -> Bool) {
            let texts = cases.filter { pred($0.name) }.map { $0.a }
            guard !texts.isEmpty else { return }
            let bytes = texts.reduce(0) { $0 + $1.utf8.count }
            var best = Double.infinity
            var ntok = 0
            for _ in 0..<3 {
                ntok = 0
                let a = nowNs()
                for t in texts { ntok += tok.encode(t).count }
                best = min(best, Double(nowNs() - a) / 1e9)
            }
            var dbest = Double.infinity
            let encs = texts.map { tok.encode($0) }
            for _ in 0..<3 {
                let a = nowNs()
                for e in encs { blackHole(tok.decode(e)) }
                dbest = min(dbest, Double(nowNs() - a) / 1e9)
            }
            print("\(label.padding(toLength: 22, withPad: " ", startingAt: 0)) \(f1(Double(bytes) / 1e6)) MB  encode \(f1(Double(bytes) / 1e6 / best)) MB/s  \(f2(Double(ntok) / 1e6 / best)) Mtok/s   decode \(f1(Double(bytes) / 1e6 / dbest)) MB/s")
        }
        let code: Set = ["py-stdlib", "py-research", "c-cpp", "metal", "swift", "js-ts-web", "shell", "config",
                         "py-stdlib-file", "c-cpp-file", "js-ts-web-file", "shell-file", "config-file", "metal-file",
                         "swift-file", "py-research-file", "snippet", "snippet-crlf"]
        group("code") { code.contains($0) }
        group("markdown/docs") { $0.hasPrefix("markdown") }
        group("localized (41 langs)") { $0.hasPrefix("localized") }
        group("cjk (zh/ja/ko)") { ["localized-zh_CN", "localized-zh_TW", "localized-zh_HK", "localized-ja", "localized-ko"].contains($0) }
        group("long pieces") { $0 == "long" }
        group("fuzz") { $0 == "fuzz" }
    }
}

// MARK: main

var argv = Array(CommandLine.arguments.dropFirst())
let cmd = argv.first ?? "all"
if !argv.isEmpty { argv.removeFirst() }
func arg(_ i: Int, _ d: String) -> String { i < argv.count ? argv[i] : d }

switch cmd {
case "compile":
    let a = nowNs()
    do { try Tokenizer.compile(tokenizerJSON: arg(0, defaultJSON), output: arg(1, defaultTKN)) } catch { fail("\(error)") }
    print("compiled \(arg(1, defaultTKN)) in \(f1(Double(nowNs() - a) / 1e6)) ms")
case "test":
    let tok = (try? Tokenizer(path: arg(0, defaultTKN))) ?? { fail("cannot load \(arg(0, defaultTKN))") }()
    let ok = runTest(tok, arg(1, defaultGolden))
    exit(ok && runRobustness(arg(0, defaultTKN)) ? 0 : 1)
case "bench":
    runBench(arg(0, defaultTKN), arg(1, defaultGolden))
case "coldstart":
    let system = (try? String(contentsOfFile: "bench/system.txt", encoding: .utf8)) ?? ""
    let a = nowNs()
    let tok = try! Tokenizer(path: arg(0, defaultTKN))
    let b = nowNs()
    let ids = tok.encodeChat(system: system, user: "list all files larger than 100MB in my home directory")
    let c = nowNs()
    let ids2 = tok.encodeChat(system: system, user: "show which process is listening on port 8080")
    let d = nowNs()
    print("init \(f2(Double(b - a) / 1e3)) us  first encodeChat \(f2(Double(c - b) / 1e3)) us  second \(f2(Double(d - c) / 1e3)) us  (\(ids.count)+\(ids2.count) tokens)")
case "encode":
    let tok = try! Tokenizer(path: defaultTKN)
    let specials = argv.first == "--specials"
    for s in argv where s != "--specials" {
        let ids = specials ? tok.encodeWithSpecials(s) : tok.encode(s)
        print(ids.map(String.init).joined(separator: " "))
    }
case "all":
    let a = nowNs()
    do { try Tokenizer.compile(tokenizerJSON: defaultJSON, output: defaultTKN) } catch { fail("\(error)") }
    print("compiled \(defaultTKN) in \(f1(Double(nowNs() - a) / 1e6)) ms")
    let tok = try! Tokenizer(path: defaultTKN)
    let ok = runTest(tok, defaultGolden) && runRobustness(defaultTKN)
    runBench(defaultTKN, defaultGolden)
    exit(ok ? 0 : 1)
default:
    fail("unknown command \(cmd)")
}
