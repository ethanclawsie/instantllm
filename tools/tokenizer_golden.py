#!/usr/bin/env python3
"""Golden encodings for `tokcheck`, produced with HuggingFace `tokenizers` (the reference).

Modes (one record per case):
  0 plain       encode(text) with NO added-token matching   (Swift: Tokenizer.encode)
  1 specials    HF Tokenizer.encode(text, add_special_tokens=False)   (Swift: encodeWithSpecials)
  2 chat        HF encode of the SPEC chat template rendered with (system, user)   (Swift: encodeChat)
  3 decode      HF decode(ids, skip_special_tokens=False) of arbitrary id sequences   (Swift: decode)
  4 nfc         HF NFC normalizer output   (Swift: normalize)
For modes 0-2 the expected string is HF decode(ids) (== NFC(text), checked here).

Corpus: benchmark prompts + chat-templated strings, llama.cpp tokenizer test inputs, hand-written
edge cases (contractions incl. case folds, whitespace runs, CRLF, tabs, NBSP/ideographic space,
digits, punctuation, paths, URLs, shell, code, CJK, emoji/ZWJ, NFC-sensitive accents, Hangul jamo,
RTL, many scripts, control chars, added tokens and fragments of them), exhaustive short strings over
a tricky alphabet, random fuzz, long pieces (O(n log n) BPE path), a per-code-point sweep of every
BMP code point and sampled supplementary ones in several contexts, and real files found on this Mac
(Python stdlib, llama.cpp / mlx / mlx-lm sources and docs, Homebrew packages).

File format (little endian): b"TKGOLD01", u32 count, then per case:
  u8 mode, u8 name_len, name, u32 a_len, a, u32 b_len, b, u32 n_ids, i32[n_ids], u32 exp_len, exp

Run: ~/Development/instantllm/.venv/bin/python -I tools/tokenizer_golden.py [out.bin]
"""
import glob
import json
import os
import random
import struct
import sys
import sysconfig
import time
import unicodedata

from tokenizers import Tokenizer, normalizers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, "models/hf/tokenizer.json")
STRESS = "--stress" in sys.argv  # one-off big run: different seed, whole stdlib, every code point
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = _pos[0] if _pos else os.path.join(ROOT, "scratch/tok/stress.bin" if STRESS else "scratch/tok/golden.bin")
TEMPLATE = "<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
rng = random.Random(777 if STRESS else 20261010)


def log(*a):
    print(*a, flush=True)


tok = Tokenizer.from_file(TOK_JSON)
_j = json.load(open(TOK_JSON, encoding="utf-8"))
_j["added_tokens"] = []
tok_plain = Tokenizer.from_str(json.dumps(_j))
NFC = normalizers.NFC()
SPECIALS = [t["content"] for t in json.load(open(TOK_JSON, encoding="utf-8"))["added_tokens"]]

cases = []  # (mode, name, a, b)


def add(mode, name, a, b=""):
    cases.append((mode, name, a, b))


def add_text(name, s, specials=True):
    add(0, name, s)
    if specials:
        add(1, name, s)


# ---------------------------------------------------------------- benchmark prompts
system = open(os.path.join(ROOT, "bench/system.txt"), encoding="utf-8").read()
prompts = [l for l in open(os.path.join(ROOT, "bench/prompts.txt"), encoding="utf-8").read().split("\n") if l]
add_text("bench-system", system)
for p in prompts:
    add_text("bench-prompt", p)
    add(2, "bench-chat", system, p)
    add(1, "bench-rendered", TEMPLATE.format(s=system, u=p))
try:
    from transformers import AutoTokenizer  # cross-check the template against the model's jinja template
    at = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/hf"))
    for p in prompts:
        r = at.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)
        assert r == TEMPLATE.format(s=system, u=p), "chat template mismatch"
    log("chat template matches transformers apply_chat_template for all prompts")
except ImportError:
    log("transformers not available: template cross-check skipped")

# ---------------------------------------------------------------- llama.cpp tokenizer tests
for f in sorted(glob.glob(os.path.join(ROOT, "scratch/research/llama.cpp/models/*.inp"))):
    for x in open(f, encoding="utf-8").read().split("\n__ggml_vocab_test__\n"):
        add_text("llamacpp-inp", x)

# ---------------------------------------------------------------- hand-written edge cases
EDGE = [
    "", " ", "  ", "   ", "\n", "\r\n", "\r", "\t", "\v", "\f", "a", "1", "'", "''", "'''",
    # contractions (incl. case-insensitive and Unicode case folds)
    "'s", "'S", "'ſ", "'t", "'T", "'re", "'RE", "'Re", "'rE", "'ve", "'VE", "'vE", "'m", "'M", "'ll", "'LL",
    "'lL", "'Ll", "'d", "'D", "'x", "'r", "'l", "'lx", "'rex", "'velvet", "'dog", "'Sun", "'mama", "'tis",
    "it's", "It'S", "IT'S", "he'll", "HE'LL", "she'd", "we're", "WE'RE", "they've", "I'm", "don't", "DON'T",
    "y'all", "rock'n'roll", "'s's's", " 's", "  's", "\t's", "\n's", "a's", "1's", "!'s", "''s", "'1", "' s",
    "it\u2019s", "\u2018quoted\u2019", "it'ſ fine", "'\u212a", "'\u017ft", "'\u0130", "'re'", "''re", "'ll've",
    "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL",
    # numbers
    "0", "12", "123", "1234567890", "3.14159", "-42", "+1e-10", "1,000,000", "0x1F", "0b1010", "v1.2.3",
    "100%", "$9.99", "\u00b2\u00b3", "\u00bd", "\u216b", "\u0661\u0662\u0663", "\u0967\u0968\u0969", "\u4e00\u4e8c\u4e09",
    "\u2460\u2461\u2462", "\U0001d7d9\U0001d7da", "\uff11\uff12\uff13", "a1b2c3", "x264", "2024-01-15T10:30:00Z",
    "192.168.1.1", "::1", "fe80::1%lo0", "1e9", "1_000_000", "3/4", "10:30pm", "#123", "abc123def456",
    # whitespace
    "a  b", "a   b", "a \n b", "a\n\nb", "a\r\n\r\nb", "a\t\tb", "\t\ta", "  \n  x", " \n", "\n ", "x \n ",
    "\u00a0a", "a\u00a0\u00a0b", "\u3000a", "a\u3000\u3000b", "a\u2028b", "a\u2029b", "a\u0085b", "a\u000bb",
    "a\u000cb", "\u200ba", "\ufeffa", "abc   ", "   abc", " \t \n \t ", "\n\n\n", "\r\r\n", "a\rb", "\n\r",
    "a \u00a0b", "a\u00a0 b", " \u3000 x", "\u1680x", "\u2000\u2001\u2002x", "\u202fx", "\u205fx",
    "x\x1cy\x1dz\x1e\x1f", "a \t1", "a  1", "a\t 1", " 1", "  1", "\t1", "\n1", " !", "  !", "\t!", "\n!",
    "foo  \n  bar", "foo \r\n bar", "line1\r\nline2\r\n", "trailing \n", "  \n\n  \n", "\t\n\t\n",
    # punctuation and code-ish
    "!!!", "...", "a...b", "(foo)", "[bar]", "{baz}", "foo()", "x+=1", "a->b", "a=>b", "a::b", "a//b",
    "#!/bin/bash", "<<EOF", "&&", "||", "|", ";;", "@decorator", "$HOME", "${VAR}", "~/", "%s", "\\n",
    "\"quoted\"", "`cmd`", "!!\n\n", " !\n", ".\r\n", "==\n\n\n", "?!\r\r", "...\n  ", "(\n", " (\n",
    "if (x) {\n\treturn y;\n}\n", "def f(x):\n    return x * 2\n", "for i in range(10):\n\tprint(i)\r\n",
    "SELECT * FROM users WHERE id = 1;", "<div class=\"a\">b</div>", "a < b && c > d", "x<y>z", "<>", "</>",
    # shell commands, paths, URLs
    "ls -la ~/Documents | grep '.pdf'", "find . -name '*.py' -mtime -7", "lsof -i :8080",
    "tar -czvf logs.tar.gz logs/", "git log --oneline --graph --all", "/usr/local/bin/python3",
    "C:\\Users\\name\\file.txt", "~/Library/Application Support/", "https://example.com/path?query=1&b=2#frag",
    "http://localhost:8080/api/v1/users", "user@example.com", "ftp://ftp.example.org/pub/file.tar.gz",
    "file:///tmp/x", "sudo rm -rf /var/tmp/*", "echo $((1+2))", "awk '{print $1}' file | sort | uniq -c",
    "ffmpeg -i video.mov -vcodec h264 -acodec aac output.mp4", "curl -fsSL https://x.sh | sh",
    "du -sh * | sort -hr", "kill -9 $(pgrep node)", "sed -i '' 's/foo/bar/g' *.txt",
    # CJK, emoji
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4.", "\u4e2d\u6587English\u6df7\u5408", "\u6f22\u5b57\u304b\u306a\u30ab\u30ca",
    "\U0001f600", "\U0001f44d\U0001f3fd", "\U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466",
    "\U0001f3f3\ufe0f\u200d\U0001f308", "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5", "1\ufe0f\u20e3", "\u2764\ufe0f",
    "\U0001f9d1\U0001f3fb\u200d\U0001f4bb", "emoji\U0001f600in text", " \U0001f600", "\U0001f600\U0001f600\U0001f600",
    "\U0001f600word", "word\U0001f600", "#\ufe0f\u20e3", "\U0001faf1\U0001f3fc\u200d\U0001faf2\U0001f3ff",
    # accents, combining marks, NFC
    "caf\u00e9", "cafe\u0301", "e\u0301\u0302", "a\u0323\u0301", "a\u0301\u0323", "\u1100\u1161\u11a8",
    "\u1100\u1161", "\uac00\u11a8", "\u212b", "\u2126", "\u0958", "\u00c5", "\ufb01", "\u01d6", "\u1e69",
    "q\u0307\u0323", "\u0344", "\u0f73", "e\u0301 e\u0301", "\u00c4\u0301", "\u017f\u0307", "\u212a", "o\u0308\u0304",
    "A\u030a", "\u0041\u0308\u0301", "\u1e9b\u0323", "\u0301abc", " \u0301", "\u0301\u0301", "x\u0301\u0301y",
    "Cu\u031b\u0309a Vi\u00ea\u0323t", "\u00e0" * 50, "a\u0300" * 50, "\u0915\u093c", "\u09c7\u09be", "\u0b47\u0b3e",
    "\u0bc6\u0bbe", "\u0cbf\u0cd5", "\u0d46\u0d3e", "\u0dd9\u0dcf", "\u1025\u102e", "\u1b05\u1b35", "\U00011099\U000110ba",
    "\u05e9\u05c1\u05bc", "\u05e9\u05bc\u05c1", "\u0627\u0653", "\u0648\u0654", "\u0627\u0655",
    # RTL and other scripts
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
    "\u0639\u0631\u0628\u064a 123 English", "\u05d1\u05b0\u05bc\u05e8\u05b5\u05d0\u05e9\u05b4\u05c1\u05d9\u05ea",
    "\u202eRTL override\u202c", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1 \u03ba\u03cc\u03c3\u03bc\u03b5",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!", "\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0",
    "\u0532\u0561\u0580\u0565\u0582", "\u1230\u120b\u121d", "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd",
    "\u09a8\u09ae\u09b8\u09cd\u0995\u09be\u09b0", "\u179f\u17bd\u179f\u17d2\u178f\u17b8", "\u1019\u1004\u103a\u1002\u101c\u102c\u1015\u102b",
    "\u0f56\u0f40\u0fb2\u0f0b\u0f64\u0f72\u0f66\u0f0b", "\u0d28\u0d2e\u0d38\u0d4d\u0d15\u0d3e\u0d30\u0d02",
    # math, fullwidth, misc symbols
    "\u2200x\u2208\u211d: x\u00b2\u22650", "\u2211_{i=1}^{n} i", "\u03b1+\u03b2=\u03b3", "\u221a2\u22481.414", "\u221e",
    "\uff21\uff22\uff23\uff11\uff12\uff13\uff01", "\uff76\uff80\uff76\uff85", "\u00a9\u00ae\u2122", "\u2190\u2192\u2191\u2193",
    "\u2014em\u2014dash\u2013", "\u00abguillemets\u00bb", "\u201cquotes\u201d", "\u2026", "\u00b7\u2022",
    # control characters
    "\x00", "a\x00b", "\x1b[31mred\x1b[0m", "\x7f", "\x01\x02", "\x08\x07", "a\x0bb\x0cc",
    # added tokens and fragments (mode 1 matches them atomically; mode 0 must not)
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>user\nhi<|im_end|>", "<|im_start", "im_start|>",
    "<|im_start|><|im_end|>", "<<|im_end|>>", "<|IM_END|>", "a<|endoftext|>b", " <|im_end|> ",
    "<tool_call>{\"name\": \"x\"}</tool_call>", "<|fim_prefix|>def f(x):<|fim_suffix|>\n    return y<|fim_middle|>",
    "<|vision_start|><|image_pad|><|vision_end|>", "<|box_start|>(1,2)<|box_end|>",
    "<|repo_name|>org/repo\n<|file_sep|>main.py\nprint(1)", "<|object_ref_start|>a<|object_ref_end|>",
    "<|quad_start|><|quad_end|><|vision_pad|><|video_pad|><|fim_pad|>", "<|endoftext|>" * 3, "<" * 5, "<|", "|>",
    "e<|im_end|>\u0301", "\u0301<|im_start|>", "caf\u0065\u0301<|im_end|>\u0301x", "<|im_end|>\n\n<|im_start|>",
    "x<tool_call>", "</tool_call>y", "<tool_call", "<|im_start|>system\n" + system + "<|im_end|>",
    "<|<|im_end|>|>", "<|im_end|im_end|>", "  <|endoftext|>  ", "\n<|im_start|>assistant\n",
]
for s in EDGE:
    add_text("edge", s)
add_text("edge-all", "".join(EDGE))
add_text("edge-all-nl", "\n".join(EDGE))
add_text("edge-all-sp", " ".join(EDGE))
for s in EDGE[::3]:
    add(2, "chat-edge", system, s)
    add(2, "chat-edge", s, "list files")
add(2, "chat-edge", "", "")
add(2, "chat-edge", " ", "\n")

# ---------------------------------------------------------------- long pieces
LONG = ["a" * 1000, "ab" * 500, "abc" * 400, " " * 1000, "\n" * 300, "=" * 500, "-=" * 300, "1" * 200,
        "\U0001f600" * 100, "\u4f60" * 1000, "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(2000)),
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/") for _ in range(4000)),
        "".join(rng.choice("0123456789abcdef") for _ in range(1000)), "a b " * 500, " \t" * 300, "\r\n" * 200,
        "\u00e0" * 300, "a\u0301" * 300, "\u0915\u094d" * 200, "\u3000" * 100 + "x", " " * 999 + "x",
        "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(3000)), "\U0001f469\u200d\U0001f4bb" * 50,
        "".join(rng.choice("\u0430\u0431\u0432\u0433\u0434\u0435\u0436") for _ in range(500)), "'s" * 300, "!" * 129,
        "x" * 127, "x" * 128, "x" * 129, " " * 127, " " * 128, " " * 129, "#" * 64, "#" * 65, "_" * 70]
for s in LONG:
    add_text("long", s)

# ---------------------------------------------------------------- exhaustive short strings
A20 = ["a", "Z", "s", "l", "e", "R", "1", "!", "'", " ", "\t", "\n", "\r", "\u00a0", "\u00e9", "\u0663",
       "\u2014", "\u0301", "\u017f", "\U0001f600"]
A14 = ["a", "s", "l", "E", "1", "!", "'", " ", "\t", "\n", "\r", "\u3000", "\u00e9", "\u0301"]


def all_strings(alpha, n):
    if n == 0:
        yield ""
        return
    for p in all_strings(alpha, n - 1):
        for c in alpha:
            yield p + c


for n in (1, 2, 3):
    for s in all_strings(A20, n):
        add(0, f"exh20-{n}", s)
for s in all_strings(A14, 4):
    add(0, "exh14-4", s)
for _ in range(40000):
    add(0, "rand20", "".join(rng.choice(A20) for _ in range(rng.randint(4, 12))))

# ---------------------------------------------------------------- fuzz
WS = [" ", " ", " ", "\t", "\n", "\r\n", "\r", "\u00a0", "\u3000", "\u2028", "\u0085", "\v", "\f", "  ", "\n\n"]
UNI = ("\u00e9\u00e8\u00fc\u00f1\u00e7\u00df\u0153\u00e6\u00f8\u00e5\u0101\u0107\u0142\u017e\u0219\u0103"
       "\u03b1\u03b2\u03b3\u03a9\u0436\u0449\u044f\u05d0\u05d1\u0627\u0628\u0916\u0917\u093f\u0e01\u0e32"
       "\u4e00\u4e2d\u6587\u65e5\u672c\u3042\u30a2\ud55c\uae00\u0661\u0662\u0967\u2460\u00bd\u00b2"
       "\u0301\u0300\u0308\u0323\u0327\u05bc\u064b\u093c\u0e31\u3099\u1161\u11a8\u1100"
       "\u2014\u2013\u2026\u201c\u201d\u2018\u2019\u00ab\u00bb\u20ac\u00a3\u00a5\u00a9\u2122\u2192\u2264\u221e"
       "\u017f\u212a\u212b\u2126\ufb01\u200b\u200d\ufe0f\ufeff")
EMO = ["\U0001f600", "\U0001f44d\U0001f3fd", "\U0001f469\u200d\U0001f4bb", "\u2764\ufe0f", "\U0001f1fa\U0001f1f8",
       "\U0001f680", "\U0001f999", "\u2705"]
FRAG = SPECIALS + ["<|im_", "_end|>", "<|", "|>", "<", ">", "<tool", "call>", "'s", "'ll", "'RE", "'Ve", "'d", "'M"]
ASCII = [chr(c) for c in range(32, 127)]


def fuzz_char():
    r = rng.random()
    if r < 0.50:
        return rng.choice(ASCII)
    if r < 0.65:
        return rng.choice(WS)
    if r < 0.85:
        return rng.choice(UNI)
    if r < 0.90:
        return rng.choice(EMO)
    if r < 0.97:
        return rng.choice(FRAG)
    return chr(rng.choice([0, 1, 7, 8, 0x1b, 0x1c, 0x7f, 0x9f, 0xad]))


for _ in range(300000 if STRESS else 30000):
    add_text("fuzz", "".join(fuzz_char() for _ in range(rng.randint(1, 60))))
if STRESS:  # random strings over all assigned code points, with ASCII/whitespace glue
    ASSIGNED = [c for c in range(0x80, 0x110000) if not (0xD800 <= c <= 0xDFFF) and unicodedata.category(chr(c)) != "Cn"]
    for _ in range(100000):
        add_text("fuzz-allcp", "".join(chr(rng.choice(ASSIGNED)) if rng.random() < 0.5 else fuzz_char()
                                       for _ in range(rng.randint(1, 40))))

# ---------------------------------------------------------------- per-code-point sweep
def assigned(c):
    return unicodedata.category(chr(c)) != "Cn"


sweep = [c for c in range(0x80, 0x10000) if not (0xD800 <= c <= 0xDFFF)]
for c in range(0x10000, 0x110000):
    if STRESS:
        sweep.append(c)
    elif 0x20000 <= c <= 0x3FFFF or 0xF0000 <= c:  # CJK ext / private use: sample
        if c % 97 == 0:
            sweep.append(c)
    elif assigned(c) or c % 13 == 0:
        sweep.append(c)
for k in range(0, len(sweep), 16):
    s = "".join("a" + chr(c) + " " + chr(c) + "b\t" + chr(c) + "1" + chr(c) + chr(c) + "'" + chr(c) + "\n"
                for c in sweep[k:k + 16])
    add(0, "cp-sweep", s)
log(f"code point sweep: {len(sweep)} code points")

# ---------------------------------------------------------------- real files
def files(patterns, roots, exclude=("/.build/", "/node_modules/", "/.git/")):
    out = []
    for r in roots:
        for pat in patterns:
            out += [p for p in glob.glob(os.path.join(r, "**", pat), recursive=True)
                    if os.path.isfile(p) and not any(e in p for e in exclude)]
    return sorted(set(out))


def read_text(p):
    try:
        with open(p, encoding="utf-8", newline="") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def add_files(name, paths, line_budget, chunk_lines=60, whole_file_limit=200000):
    rng.shuffle(paths)
    lines_used, nfiles = 0, 0
    for p in paths:
        if lines_used >= line_budget:
            break
        t = read_text(p)
        if not t or len(t) > 2_000_000:
            continue
        nfiles += 1
        ls = t.split("\n")
        if len(t) <= whole_file_limit and nfiles % 10 == 1:
            add_text(name + "-file", t)
        for k in range(0, len(ls), chunk_lines):
            chunk = "\n".join(ls[k:k + chunk_lines]) + ("\n" if k + chunk_lines < len(ls) else "")
            add_text(name, chunk)
        lines_used += len(ls)
    log(f"{name}: {nfiles} files, {lines_used} lines")


STD = sysconfig.get_paths()["stdlib"]
RES = os.path.join(ROOT, "scratch/research")
HB = ["/opt/homebrew/Cellar", "/opt/homebrew/lib", "/opt/homebrew/share"]
add_files("py-stdlib", files(["*.py"], [STD]), 10**7 if STRESS else 40000)
add_files("py-research", files(["*.py"], [RES]), 8000)
add_files("markdown", files(["*.md", "*.rst", "README*", "*.txt"], [RES] + HB + [STD]), 10**6 if STRESS else 25000)
add_files("c-cpp", files(["*.c", "*.cpp", "*.h", "*.hpp", "*.cc", "*.m", "*.mm"], [RES]), 20000)
add_files("metal", files(["*.metal"], [RES]), 4000)
add_files("swift", files(["*.swift"], [RES, os.path.join(ROOT, "Sources")] + HB), 6000)
add_files("js-ts-web", files(["*.js", "*.ts", "*.tsx", "*.html", "*.css", "*.vue", "*.svelte"], [RES] + HB), 8000)
add_files("shell", files(["*.sh", "*.bash", "*.zsh", "Makefile", "*.mk"], [RES] + HB), 5000)
add_files("config", files(["*.json", "*.yaml", "*.yml", "*.toml", "CMakeLists.txt", "*.cmake"], [RES] + HB), 8000)
add_files("rust-go", files(["*.rs", "*.go"], [RES] + HB), 5000)


def localized_strings(line_budget_per_lang=1500, chunk_lines=30):
    """Real UI text in ~40 languages from macOS localized .strings files (binary plists)."""
    import plistlib
    apps = ["/System/Library/CoreServices/Finder.app"] + sorted(glob.glob("/System/Applications/*.app"))
    langs = sorted({os.path.basename(d)[:-6] for d in glob.glob(apps[0] + "/Contents/Resources/*.lproj")})
    total = 0
    for lang in langs:
        lines = []
        for app in apps:
            for f in sorted(glob.glob(f"{app}/Contents/Resources/{lang}.lproj/*.strings")):
                try:
                    d = plistlib.load(open(f, "rb"))
                except Exception:
                    continue
                if isinstance(d, dict):
                    lines += [v for v in d.values() if isinstance(v, str)]
                if len(lines) >= line_budget_per_lang:
                    break
            if len(lines) >= line_budget_per_lang:
                break
        lines = lines[:line_budget_per_lang]
        total += len(lines)
        for k in range(0, len(lines), chunk_lines):
            add_text("localized-" + lang, "\n".join(lines[k:k + chunk_lines]))
    log(f"localized: {len(langs)} languages, {total} lines")


localized_strings(10**5 if STRESS else 1500)

SNIPPETS = [
    'fn main() {\n    let v: Vec<u32> = (1..=10).map(|x| x * x).collect();\n    println!("{:?}", v);\n}\n',
    'impl<T: Clone + Send> Foo<T> for Bar<\'a, T> where T: \'static {\n    fn f(&self) -> Option<&T> { self.0.get(0) }\n}\n',
    'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\tif len(os.Args) < 2 {\n\t\tfmt.Println("usage")\n\t}\n}\n',
    'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello, " + args[0]);\n    }\n}\n',
    'fun main() = runBlocking { val xs = listOf(1, 2, 3).map { it * 2 }; println(xs) }',
    'class Foo < Bar\n  def initialize(x) = @x = x\n  attr_reader :x\nend\nputs Foo.new(1).x\n',
    '<?php\n$arr = ["a" => 1, "b" => 2];\nforeach ($arr as $k => $v) { echo "$k=$v\\n"; }\n',
    'main :: IO ()\nmain = mapM_ print $ take 10 $ filter even [1..]\n',
    '(defun fact (n) (if (<= n 1) 1 (* n (fact (- n 1)))))',
    'local t = {1, 2, 3}\nfor i, v in ipairs(t) do print(i, v) end\n',
    'my %h = (a => 1); while (my ($k, $v) = each %h) { print "$k $v\\n"; }',
    'df <- read.csv("data.csv"); summary(lm(y ~ x, data = df))',
    'function f(x::Vector{Float64})::Float64\n    return sum(x .^ 2)\nend\n',
    '_start:\n    mov rax, 60\n    xor edi, edi\n    syscall\n',
    '\\begin{equation}\n  E = mc^2 \\quad \\text{and} \\quad \\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}\n\\end{equation}\n',
    'FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD ["python", "app.py"]\n',
    'services:\n  web:\n    image: "nginx:latest"\n    ports:\n      - "8080:80"\n',
    'SELECT u.name, COUNT(o.id) AS n FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.name HAVING n > 5;',
    'const f = async (x) => { const r = await fetch(`/api/${x}`); return r.ok ? r.json() : null; };',
    'template <typename T, std::size_t N>\nconstexpr auto sum(const std::array<T, N>& a) -> T { T s{}; for (auto& x : a) s += x; return s; }\n',
    'kernel void add(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] += 1.0f; }',
    'let x = try await withThrowingTaskGroup(of: Int.self) { g in for i in 0..<4 { g.addTask { i * i } }; return try await g.reduce(0, +) }',
]
for sn in SNIPPETS:
    add_text("snippet", sn)
    add_text("snippet-crlf", sn.replace("\n", "\r\n"))

# ---------------------------------------------------------------- decode cases
for i in range(151936):
    add(3, "decode-single", "", "")  # ids filled below
n_single = 151936
rand_ids = []
pool_partial = list(range(0, 256)) + rng.sample(range(256, 151643), 3000) + list(range(151643, 151665))
for _ in range(20000):
    rand_ids.append([rng.choice(pool_partial) for _ in range(rng.randint(1, 8))])
for ids in rand_ids:
    add(3, "decode-random", "", "")
rand_ids += [[], [151936], [-1 & 0x7FFFFFFF]]
add(3, "decode-random", "", "")
add(3, "decode-random", "", "")
add(3, "decode-random", "", "")

# ---------------------------------------------------------------- NFC cases
NFD = normalizers.NFD()
nfc_single = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF) and NFC.normalize_str(chr(c)) != chr(c)]
for c in nfc_single:
    add(4, "nfc-single", chr(c))
pool = [c for c in range(0x300, 0x110000) if not (0xD800 <= c <= 0xDFFF) and unicodedata.combining(chr(c))]
pool += [ord(x) for x in "aeiouAEIOUnNcCsSzZyYwW "] + list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176))
pool += list(range(0x11A8, 0x11C3)) + nfc_single[:400] + [0xAC00, 0xAC01, 0xD7A3, 0x0B47, 0x0B3E, 0x0BC6, 0x0BBE]
for _ in range(30000):
    add(4, "nfc-random", "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 10))))
for _ in range(5000):
    s = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 10)))
    add_text("nfc-random-enc", s)

# ---------------------------------------------------------------- compute expectations
log(f"{len(cases)} cases; encoding with HF ...")
t0 = time.time()
idx = {m: [i for i, c in enumerate(cases) if c[0] == m] for m in range(5)}
results = [None] * len(cases)
BATCH = 20000


def run_batch(indices, fn_texts, encoder):
    for k in range(0, len(indices), BATCH):
        part = indices[k:k + BATCH]
        encs = encoder.encode_batch([fn_texts(cases[i]) for i in part], add_special_tokens=False)
        for i, e in zip(part, encs):
            results[i] = e.ids


run_batch(idx[0], lambda c: c[2], tok_plain)
run_batch(idx[1], lambda c: c[2], tok)
run_batch(idx[2], lambda c: TEMPLATE.format(s=c[2], u=c[3]), tok)
single_i = 0
rand_i = 0
for i in idx[3]:
    if cases[i][1] == "decode-single":
        results[i] = [single_i]
        single_i += 1
    else:
        results[i] = rand_ids[rand_i]
        rand_i += 1
assert single_i == n_single and rand_i == len(rand_ids)
log(f"encoded in {time.time()-t0:.1f}s")

# cross-check plain mode against the manual pipeline on a sample
for i in rng.sample(idx[0], 3000):
    s = cases[i][2]
    ids = []
    for piece, _ in tok.pre_tokenizer.pre_tokenize_str(NFC.normalize_str(s)):
        ids += [t.id for t in tok.model.tokenize(piece)]
    assert ids == results[i], f"plain pipeline mismatch for {s!r}"

out = bytearray(b"TKGOLD01")
out += struct.pack("<I", len(cases))
dec_inputs = [results[i] for i in range(len(cases)) if cases[i][0] != 4]
decoded = tok.decode_batch(dec_inputs, skip_special_tokens=False)
di = 0
n_ids = 0
for i, (mode, name, a, b) in enumerate(cases):
    if mode == 4:
        ids = []
        exp = NFC.normalize_str(a)
    else:
        ids = results[i]
        exp = decoded[di]
        di += 1
        if mode in (0, 1):
            assert exp == NFC.normalize_str(a), f"HF round trip != NFC for {a!r}"
    n_ids += len(ids)
    nb = name.encode()
    ab, bb, eb = a.encode("utf-8"), b.encode("utf-8"), exp.encode("utf-8")
    out += struct.pack("<BB", mode, len(nb)) + nb
    out += struct.pack("<I", len(ab)) + ab + struct.pack("<I", len(bb)) + bb
    out += struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}i", *ids)
    out += struct.pack("<I", len(eb)) + eb
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(out)
counts = {}
for c in cases:
    counts[c[1]] = counts.get(c[1], 0) + 1
log(f"wrote {OUT}: {len(cases)} cases, {n_ids} ids, {len(out)/1e6:.1f} MB")
log("categories: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
