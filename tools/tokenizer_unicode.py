#!/usr/bin/env python3
"""Derive the Unicode tables the Swift tokenizer needs, empirically, from HF `tokenizers` itself.

HF tokenizers (0.23.x) uses Oniguruma for the Qwen2 split regex and the old
`unicode-normalization-alignments 0.1.12` crate for NFC. Both bundle their own (different, older)
Unicode versions, so instead of trusting Python's / Swift's / ICU's Unicode data we probe HF for
every code point:

  * pre-tokenizer class: L (\\p{L}), N (\\p{N}), WS (\\s minus space/CR/LF), SPACE, CRLF, O (other)
  * which letters the case-insensitive contractions (?i:'s|'t|'re|'ve|'m|'ll|'d) accept
  * canonical decomposition (crate NFD of each code point), canonical combining class (by probing
    canonical reordering), primary compositions (verified with crate NFC)

A Python prototype of the NFC algorithm the Swift code implements is validated against HF NFC on
all single code points, all decompositions and many random sequences before anything is emitted.

Output: Sources/Tokenizer/UnicodeData.swift (a base64 blob consumed by Tokenizer.compile).

Run: ~/Development/instantllm/.venv/bin/python -I tools/tokenizer_unicode.py
"""
import base64
import os
import random
import struct
import sys
import time
import unicodedata

from tokenizers import Tokenizer, normalizers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, "models/hf/tokenizer.json")
OUT_SWIFT = os.path.join(ROOT, "Sources/Tokenizer/UnicodeData.swift")

CLS_L, CLS_N, CLS_O, CLS_WS, CLS_SPACE, CLS_CRLF = 0, 1, 2, 3, 4, 5
F_NFC = 1 << 3      # code point may change under NFC (in some context): take the slow path
F_C1 = 1 << 4       # after "'": s/t/m/d (any case-fold)  -> 2-char contraction
F_RV = 1 << 5       # after "'": r/v  (needs F_E next)     -> 3-char contraction
F_E = 1 << 6        # second char of 're / 've
F_LL = 1 << 7       # both chars of 'll

SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

ALL = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]


def log(*a):
    print(*a, flush=True)


def main():
    t_start = time.time()
    tok = Tokenizer.from_file(TOK_JSON)
    pt = tok.pre_tokenizer
    NFD = normalizers.NFD()
    NFC = normalizers.NFC()

    def starts(s):
        return set(o[0] for _, o in pt.pre_tokenize_str(s))

    def boundary_probe(cands, make, offsets, chunk=20000):
        """For each candidate, build a probe string; return {c: tuple(bool boundary at each offset)}."""
        res = {}
        for k in range(0, len(cands), chunk):
            part = cands[k:k + chunk]
            buf, base, p = [], [], 0
            for c in part:
                s = make(c)
                buf.append(s)
                base.append(p)
                p += len(s)
            st = starts("".join(buf))
            for c, b in zip(part, base):
                res[c] = tuple((b + o) in st for o in offsets)
        return res

    # ---------------- pre-tokenizer classes ----------------
    p1 = boundary_probe(ALL, lambda c: "\na" + chr(c), [2])
    letters = [c for c in ALL if not p1[c][0]]
    nonL = [c for c in ALL if p1[c][0]]
    p2 = boundary_probe(nonL, lambda c: "\n" + chr(c) + "a", [2])
    rest = [c for c in nonL if not p2[c][0]]
    p3 = boundary_probe(rest, lambda c: "\n!" + chr(c), [2])
    cls = {}
    for c in letters:
        cls[c] = CLS_L
    for c in nonL:
        if p2[c][0]:
            cls[c] = CLS_CRLF if c in (10, 13) else CLS_N
        else:
            cls[c] = CLS_WS if p3[c][0] else CLS_O
    cls[0x20] = CLS_SPACE
    assert cls[10] == CLS_CRLF and cls[13] == CLS_CRLF
    ws = [c for c in ALL if cls[c] in (CLS_WS, CLS_SPACE, CLS_CRLF)]
    log(f"classes: L={len(letters)} N={sum(1 for c in ALL if cls[c]==CLS_N)} "
        f"WS={len(ws)} {[hex(c) for c in ws]}  ({time.time()-t_start:.1f}s)")

    # ---------------- contraction letters ----------------
    q1 = boundary_probe(letters, lambda c: "\n'" + chr(c) + "b", [2, 3])
    q2 = boundary_probe(letters, lambda c: "\n'" + chr(c) + "eb", [2, 3, 4])
    q3 = boundary_probe(letters, lambda c: "\n'" + chr(c) + "lb", [2, 3, 4])
    q4 = boundary_probe(letters, lambda c: "\n'r" + chr(c) + "b", [2, 3, 4])
    q5 = boundary_probe(letters, lambda c: "\n'v" + chr(c) + "b", [2, 3, 4])
    q6 = boundary_probe(letters, lambda c: "\n'l" + chr(c) + "b", [2, 3, 4])
    set_c1 = [c for c in letters if q1[c] == (False, True)]
    set_rv = [c for c in letters if q2[c] == (False, False, True)]
    set_l1 = [c for c in letters if q3[c] == (False, False, True)]
    set_e_r = [c for c in letters if q4[c] == (False, False, True)]
    set_e_v = [c for c in letters if q5[c] == (False, False, True)]
    set_l2 = [c for c in letters if q6[c] == (False, False, True)]
    log("contractions: c1", "".join(map(chr, set_c1)), "rv", "".join(map(chr, set_rv)),
        "e(r)", "".join(map(chr, set_e_r)), "e(v)", "".join(map(chr, set_e_v)),
        "l1", "".join(map(chr, set_l1)), "l2", "".join(map(chr, set_l2)))
    assert set_e_r == set_e_v and set_l1 == set_l2, "contraction fold sets differ by position"

    # ---------------- NFC data (crate behaviour) ----------------
    SEP = "\x01"

    def batch_norm(norm, strs, chunk=20000):
        out = []
        for k in range(0, len(strs), chunk):
            part = strs[k:k + chunk]
            r = norm.normalize_str(SEP + SEP.join(part) + SEP)
            pieces = r.split(SEP)
            assert len(pieces) == len(part) + 2, "separator lost"
            out.extend(pieces[1:-1])
        return out

    no_sep = [c for c in ALL if c != 1]
    chars = [chr(c) for c in no_sep]
    nfd1 = dict(zip(no_sep, batch_norm(NFD, chars)))
    nfc1 = dict(zip(no_sep, batch_norm(NFC, chars)))
    nfd1[1] = nfc1[1] = "\x01"
    decomp = {c: [ord(x) for x in nfd1[c]] for c in ALL if nfd1[c] != chr(c)}
    hangul = [c for c in range(SBASE, SBASE + SCOUNT)]
    assert all(c in decomp for c in hangul), "crate does not decompose all Hangul syllables"
    log(f"crate decompositions: {len(decomp)} (incl. {len(hangul)} Hangul)  ({time.time()-t_start:.1f}s)")

    # ccc: probe canonical reordering against U+0334 (ccc 1) and U+0301 (ccc 230)
    nodec = [c for c in ALL if c not in decomp and c != 1]
    r1 = batch_norm(NFD, ["a" + chr(c) + "̴" for c in nodec])
    r2 = batch_norm(NFD, ["á" + chr(c) for c in nodec])
    moved = set()
    for c, x, y in zip(nodec, r1, r2):
        if x == "a̴" + chr(c) or y == "a" + chr(c) + "́":
            moved.add(c)
    ccc = {}
    for c in moved:
        v = unicodedata.combining(chr(c))
        assert v != 0, f"crate reorders U+{c:04X} but Python ccc is 0"
        ccc[c] = v
    py_nonzero = [c for c in ALL if unicodedata.combining(chr(c)) != 0]
    unknown = [c for c in py_nonzero if c not in moved and c not in decomp]
    log(f"crate non-starters: {len(ccc)}; Python non-starters unknown to crate: {len(unknown)}")
    # validate exact relative order against references at every ccc value
    values = sorted(set(ccc.values()))
    ref = {}
    for c in sorted(ccc):
        ref.setdefault(ccc[c], c)
    probes, expect = [], []
    for c in sorted(ccc):
        for v in values:
            r = ref[v]
            if r == c:
                continue
            probes.append("a" + chr(c) + chr(r))
            expect.append("a" + chr(r) + chr(c) if ccc[c] > v else "a" + chr(c) + chr(r))
            probes.append("a" + chr(r) + chr(c))
            expect.append("a" + chr(c) + chr(r) if ccc[c] < v else "a" + chr(r) + chr(c))
    got = batch_norm(NFD, probes)
    bad = sum(1 for g, e in zip(got, expect) if g != e)
    assert bad == 0, f"{bad} ccc ordering mismatches"
    log(f"ccc relative order validated with {len(probes)} probes")

    # compositions: Python primary composites, verified against the crate
    pairs = {}
    for c in ALL:
        d = unicodedata.decomposition(chr(c))
        if not d or d.startswith("<"):
            continue
        parts = [int(x, 16) for x in d.split()]
        if len(parts) != 2:
            continue
        if unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) != chr(c):
            continue
        if SBASE <= c < SBASE + SCOUNT:
            continue
        pairs[(parts[0], parts[1])] = c
    plist = sorted(pairs)
    got = batch_norm(NFC, [chr(a) + chr(b) for a, b in plist])
    comp = {}
    for (a, b), g in zip(plist, got):
        if g == chr(pairs[(a, b)]):
            comp[(a, b)] = pairs[(a, b)]
    log(f"compositions: python {len(pairs)}, crate-verified {len(comp)}")

    # ---------------- Python prototype of the Swift NFC ----------------
    def cccof(c):
        return ccc.get(c, 0)

    def decompose(c):
        if SBASE <= c < SBASE + SCOUNT:
            s = c - SBASE
            l, v, t = LBASE + s // NCOUNT, VBASE + (s % NCOUNT) // TCOUNT, TBASE + s % TCOUNT
            return [l, v] if t == TBASE else [l, v, t]
        return decomp.get(c, [c])

    def compose(a, b):
        if LBASE <= a < LBASE + LCOUNT and VBASE <= b < VBASE + VCOUNT:
            return SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT
        if SBASE <= a < SBASE + SCOUNT and (a - SBASE) % TCOUNT == 0 and TBASE < b < TBASE + TCOUNT:
            return a + (b - TBASE)
        return comp.get((a, b))

    def proto_nfc(s):
        cps = []
        for ch in s:
            cps.extend(decompose(ord(ch)))
        # canonical ordering: stable sort runs of non-starters
        i, n = 0, len(cps)
        while i < n:
            if cccof(cps[i]) == 0:
                i += 1
                continue
            j = i
            while j < n and cccof(cps[j]) != 0:
                j += 1
            cps[i:j] = sorted(cps[i:j], key=cccof)
            i = j
        # composition (mirrors unicode-normalization's Recompositions)
        out = []
        starter = -1           # index in out of the current composee
        last_ccc = -1          # -1: nothing since the composee
        for c in cps:
            k = cccof(c)
            if starter >= 0 and (last_ccc == -1 or last_ccc < k):
                r = compose(out[starter], c)
                if r is not None:
                    out[starter] = r
                    continue
            if k == 0:
                starter = len(out)
                last_ccc = -1
            else:
                if starter >= 0:
                    last_ccc = k
            out.append(c)
        return "".join(map(chr, out))

    # seconds of composition pairs (incl. Hangul V/T)
    seconds = set(b for _, b in comp) | set(range(VBASE, VBASE + VCOUNT)) | set(range(TBASE + 1, TBASE + TCOUNT))

    def needs_nfc(c):
        if nfc1[c] != chr(c):
            return True
        if cccof(c) != 0:
            return True
        first = decompose(c)[0]
        return first in seconds or cccof(first) != 0

    flagged = [c for c in ALL if needs_nfc(c)]
    log(f"NFC slow-path code points: {len(flagged)}; min {flagged[0]:#x}")
    assert flagged[0] >= 0x300

    # validation 1: every single code point and every decomposition
    singles = [chr(c) for c in ALL]
    mism = [c for c, g in zip(ALL, singles) if proto_nfc(g) != nfc1[c]]
    assert not mism, f"single cp NFC mismatches: {[hex(c) for c in mism[:20]]}"
    decs = ["".join(map(chr, decomp[c])) for c in sorted(decomp)]
    got = batch_norm(NFC, decs)
    mism = [d for d, g in zip(decs, got) if proto_nfc(d) != g]
    assert not mism, f"decomposition NFC mismatches: {mism[:10]}"
    # validation 2: random sequences of interesting code points
    rng = random.Random(1234)
    pool = sorted(set(flagged) | set(c for ab in comp for c in ab) | set(ccc) |
                  set(range(LBASE, LBASE + LCOUNT)) | set(rng.sample(hangul, 300)) |
                  set(map(ord, "aeiouAEIOUnNcCsSzZ ")) | set(rng.sample(unknown, min(200, len(unknown)))))
    unflagged_pool = [c for c in pool if not needs_nfc(c)] + rng.sample(
        [c for c in ALL if not needs_nfc(c)], 3000)
    rand, rand2 = [], []
    for _ in range(200000):
        rand.append("".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 10))))
    for _ in range(50000):
        rand2.append("".join(chr(rng.choice(unflagged_pool)) for _ in range(rng.randint(1, 8))))
    got = batch_norm(NFC, rand)
    mism = [s for s, g in zip(rand, got) if proto_nfc(s) != g]
    assert not mism, f"random NFC mismatches: {[s.encode('unicode_escape') for s in mism[:10]]}"
    got = batch_norm(NFC, rand2)
    mism = [s for s, g in zip(rand2, got) if g != s]
    assert not mism, f"unflagged strings changed under NFC: {[s.encode('unicode_escape') for s in mism[:10]]}"
    log(f"NFC prototype validated on {len(singles)+len(decs)+len(rand)+len(rand2)} strings "
        f"({time.time()-t_start:.1f}s)")

    # ---------------- emit ----------------
    prop = {}
    for c in ALL:
        v = cls[c] | (cccof(c) << 8)
        if needs_nfc(c):
            v |= F_NFC
        prop[c] = v
    for c in set_c1:
        prop[c] |= F_C1
    for c in set_rv:
        prop[c] |= F_RV
    for c in set_e_r:
        prop[c] |= F_E
    for c in set_l1:
        prop[c] |= F_LL
    for c in range(0xD800, 0xE000):
        prop[c] = CLS_O
    runs = []
    last = None
    for c in range(0x110000):
        if prop[c] != last:
            runs.append((c, prop[c]))
            last = prop[c]
    blob = bytearray(b"UTB1")
    blob += struct.pack("<I", len(runs))
    for c, v in runs:
        blob += struct.pack("<IH", c, v)
    dl = [(c, decomp[c]) for c in sorted(decomp) if not (SBASE <= c < SBASE + SCOUNT)]
    blob += struct.pack("<I", len(dl))
    for c, d in dl:
        blob += struct.pack("<IB", c, len(d)) + b"".join(struct.pack("<I", x) for x in d)
    cl = sorted(comp.items())
    blob += struct.pack("<I", len(cl))
    for (a, b), r in cl:
        blob += struct.pack("<III", a, b, r)
    b64 = base64.b64encode(bytes(blob)).decode()
    lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]
    with open(OUT_SWIFT, "w") as f:
        f.write("// GENERATED by tools/tokenizer_unicode.py from HF tokenizers "
                f"{__import__('tokenizers').__version__} behaviour. Do not edit.\n")
        f.write(f"// {len(runs)} property runs, {len(dl)} decompositions, {len(cl)} compositions.\n")
        f.write("// Property bits: 0-2 class (0 L, 1 N, 2 O, 3 WS, 4 SPACE, 5 CRLF), 3 NFC slow path,\n")
        f.write("// 4 contraction s/t/m/d, 5 contraction r/v, 6 contraction e, 7 contraction l, 8-15 ccc.\n\n")
        f.write("enum UnicodeData {\n    static let base64 = \"\"\"\n")
        for ln in lines:
            f.write("        " + ln + "\n")
        f.write("        \"\"\"\n}\n")
    log(f"wrote {OUT_SWIFT}: {len(blob)} bytes blob, {len(runs)} runs ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
