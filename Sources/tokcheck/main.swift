import Foundation
import Tokenizer

let args = CommandLine.arguments
let tkn = "scratch/tok/qwen.tkn"
let t0 = Date()
try Tokenizer.compile(tokenizerJSON: "models/hf/tokenizer.json", output: tkn)
print("compile", Date().timeIntervalSince(t0))
let tok = try Tokenizer(path: tkn)
for s in args.dropFirst() {
    print(tok.encodeWithSpecials(s).map(String.init).joined(separator: " "))
}
