"""ildraft: reference implementation of the instantllm speculative drafter (table format v1).

This module is the specification a Swift/Metal port must reproduce. It contains
  * Tables:      everything stored in models/draft/qwen-q4-ngram.bin, built from TRAIN outputs
  * Drafter:     per-request state + candidate generation + features + MLP scorer
  * chain/tree:  how many tokens to propose and in what shape
  * save/load:   the binary format (see FORMAT below)

Vocabulary of terms
  P      user-prompt token ids (between the 44-token system prefix and the 5-token tail)
  TAIL   [151645, 198, 151644, 77091, 198]  = "<|im_end|>\\n<|im_start|>assistant\\n"
  out    output tokens produced so far (accepted + already drafted in this step)
  ctx    TAIL + out
  y_j    training output j (generated ids + stop token if it stopped)

FORMAT (all little-endian)
  header, 64 bytes:
    char[8] magic "ILDRAFT1" | u32 version=1 | u32 n_sections
    u32 N (max n-gram order) | u32 M (next tokens kept per context) | u32 KNB (neighbours)
    u32 MAXM (max suffix match) | u32 F (features) | u32 H (MLP hidden) | u32 minc_hi
    u32 n_docs | u32 flags | u32 as_top | u32 edit_E | u32 0
  section directory: n_sections x {char[4] tag, u32 count, u64 byte offset}; sections 8-aligned
    NGKY count=C   u64[C]  context keys, ascending. key = fnv1a64(u8 order, then each token as u32 LE)
    NGVL count=C   C x {u32 total, u32 (first << 3) | n}    (parallel to NGKY; next tokens at
                                                             NGNX[first .. first+n), n <= M <= 7)
    NGNX count=X   X x {u32 token, u32 count}               sorted by count desc, token asc
    WDKY count=W   u64[W] word keys ascending: fnv1a64(word bytes), word = run of [a-z0-9]
                   after ASCII-lowercasing the prompt
    WDVL count=W   W x {f32 idf, u32 first, u32 n}           postings at POST[first .. first+n)
    POST count=Q   u32[Q] doc ids (ascending per word)
    DNRM count=D   f32[D] doc norms
    SQOF count=D+1 u32 offsets into SQTK;  SQTK count=T  u32 tokens (y_j incl. stop token)
    CPTK count=A   A x {u32 token, f32 prop}   ascending token      (default 0.2)
    CPC1 count=B   B x {u32 token, f32 rate}   ascending token      (default 0.1)
    CPC2 count=E   E x {u64 key=(t1<<32)|t2, f32 rate, u32 0} ascending key  (default 0.1)
    ASWK count=AW  u64[AW] association word keys ascending (fnv1a64(word bytes))  [flags & 1]
    ASWV count=AW  AW x {u32 first, u32 n}        pairs at ASTK[first .. first+n)
    ASTK count=AT  AT x {u32 token, f32 loglift}  per word sorted by loglift desc, token asc
    PTXO count=D+1 u32 byte offsets into PTXB  [flags & 4]
    PTXB count=nb  UTF-8 prompt texts of the training requests (for edited neighbours)
    TFLS count=TF  TF x {u32 token, u32 flags} ascending token, only tokens with flags != 0:
                   bits 0-2 = number of '`' bytes in the token text (capped 7), bit 3 = contains a
                   newline byte; every other token (and every special id >= 151643) has flags 0
    MLP0 count=n   f32[n] = W1[H][F] (row-major), b1[H], W2[H], b2[1], WL[F]
                   score(x) = sum_h W2[h]*relu(W1[h].x + b1[h]) + b2 + WL.x
"""
import heapq
import math
import re
import struct
from collections import defaultdict

TAIL = [151645, 198, 151644, 77091, 198]
STOPS = (151645, 151643)
FENCE_OPEN = (73594, 13874)      # "```", "``"
TICK = 63                        # "`"

FNV_OFF = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
M64 = (1 << 64) - 1


def fnv1a64(b: bytes) -> int:
    h = FNV_OFF
    for x in b:
        h ^= x
        h = (h * FNV_PRIME) & M64
    return h


def ctx_key(ctx_tokens) -> int:
    return fnv1a64(bytes([len(ctx_tokens)]) + struct.pack(f"<{len(ctx_tokens)}I", *ctx_tokens))


_WRE = re.compile(rb"[a-z0-9]+")
_SPACEVAR_RE = re.compile(r"[A-Za-z0-9_.\-/~:@]+")


def prompt_words(text: str):
    """Distinct words in first-appearance order: runs of [a-z0-9] after ASCII lowercasing."""
    seen = []
    s = set()
    for w in _WRE.findall(text.encode("utf-8").lower()):
        if w not in s:
            s.add(w)
            seen.append(w)
    return seen


FEATS = [
    "bias",
    "ng_p_long", "ng_top_long", "ng_ord_long", "ng_lcnt_long", "ng_p_supp3", "ng_p_supp10",
    "ng_p1", "ng_p2", "ng_p3", "ng_p4", "ng_p5", "ng_p6", "ng_rank2", "ng_rank3plus", "ng_ent_low",
    "sf_m", "sf_any", "sf_m_ge3",
    "pr_m", "pr_any", "pr_m_ge2", "pr_first",
    "nb_sum", "nb_msum", "nb_cnt", "nb_maxm", "nb_top1", "nb_top1_m",
    "is_stop", "in_prompt", "agree_ng_nb", "agree_sf_nb",
    "cp_prop", "cp_ctx1", "cp_ctx2", "cp_unused", "cp_first_unused", "cp_spacevar",
    "st_open_fence", "st_open_tick", "c_tick", "c_nl", "st_ticks_odd", "st_len", "c_repeat",
    "as_max", "as_cnt", "as_cand",
    "ed_sum", "ed_cnt", "ed_maxm", "ed_top1",
]
FI = {f: i for i, f in enumerate(FEATS)}
NF = len(FEATS)


FLAG_ASSOC = 1        # association features present (sections ASWK/ASWV/ASTK)
FLAG_ASSOC_CANDS = 2  # association tokens are also candidates
FLAG_EDIT = 4         # edited neighbours (sections PTXO/PTXB; needs the tokenizer at request start)


class Config:
    def __init__(self, N=6, M=6, KNB=16, MAXM=12, H=16, minc_hi=1, flags=FLAG_ASSOC, as_min=3, as_lift=2.0,
                 as_top=4, edit_E=4, max_df=1.0, cc2_min=1):
        self.N, self.M, self.KNB, self.MAXM, self.H, self.minc_hi = N, M, KNB, MAXM, H, minc_hi
        self.flags, self.as_min, self.as_lift, self.as_top = flags, as_min, as_lift, as_top
        self.edit_E = edit_E
        self.max_df, self.cc2_min = max_df, cc2_min   # build-time pruning only (not stored)


# ------------------------------------------------------------------------------------------
# tables

class Tables:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ng = {}        # tuple(ctx) -> (total, [(tok, cnt)...]) truncated to M
        self.vocab = {}     # word bytes -> (idf, [doc ids])
        self.norms = []
        self.seqs = []
        self.prop = {}
        self.cc1 = {}
        self.cc2 = {}
        self.W = None       # dict of numpy-free lists: W1, b1, W2, b2, WL
        self.tok_flags = None  # token -> (has_tick, has_nl); filled from a tokenizer
        self.assoc = {}     # word bytes -> [(token, loglift)] sorted by loglift desc, token asc

    # -- build --
    @staticmethod
    def build(recs, cfg: Config):
        t = Tables(cfg)
        N, M = cfg.N, cfg.M
        counts = defaultdict(lambda: defaultdict(int))
        for r in recs:
            s = TAIL + list(r["y"])
            for i in range(len(TAIL), len(s)):
                nxt = s[i]
                for n in range(1, N + 1):
                    if i - n < 0:
                        break
                    counts[tuple(s[i - n:i])][nxt] += 1
        for ctx, d in counts.items():
            tot = sum(d.values())
            if len(ctx) >= 3 and tot < cfg.minc_hi:
                continue
            top = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:M]
            t.ng[ctx] = (tot, top)
        # prompt index
        df = defaultdict(int)
        docs = []
        for r in recs:
            ws = prompt_words(r["prompt"])
            docs.append(ws)
            for w in ws:
                df[w] += 1
        D = len(recs)
        post = defaultdict(list)
        for j, ws in enumerate(docs):
            for w in ws:
                post[w].append(j)
        for w, c in df.items():
            if c > cfg.max_df * D:
                continue      # very common words carry ~no similarity signal; dropping them shrinks POST
            t.vocab[w] = (math.log((D + 1) / (c + 0.5)), post[w])
        t.norms = [math.sqrt(sum(t.vocab[w][0] ** 2 for w in ws if w in t.vocab)) or 1.0 for ws in docs]
        t.seqs = [list(r["y"]) for r in recs]
        t.ptexts = [r["prompt"] for r in recs] if cfg.flags & FLAG_EDIT else []
        # copy statistics
        inp = defaultdict(int)
        cop = defaultdict(int)
        c1 = defaultdict(lambda: [0, 0])
        c2 = defaultdict(lambda: [0, 0])
        for r in recs:
            ps = set(r["pids"])
            ys = set(r["y"])
            for x in ps:
                inp[x] += 1
                if x in ys:
                    cop[x] += 1
            s = TAIL + list(r["y"])
            for i in range(len(TAIL), len(s)):
                hit = 1 if s[i] in ps else 0
                e = c1[s[i - 1]]
                e[0] += hit
                e[1] += 1
                e = c2[(s[i - 2], s[i - 1])]
                e[0] += hit
                e[1] += 1
        t.prop = {x: (cop[x] + 0.5) / (inp[x] + 2.0) for x in inp}
        # prompt-word -> output-token association: lift = P(t in y | w in prompt) / P(t in y)
        if cfg.flags & FLAG_ASSOC:
            cw = defaultdict(int)
            ct = defaultdict(int)
            cwt = defaultdict(int)
            for r, ws in zip(recs, docs):
                ys = set(r["y"])
                for x in ys:
                    ct[x] += 1
                for w in ws:
                    cw[w] += 1
                    for x in ys:
                        cwt[(w, x)] += 1
            pairs = defaultdict(list)
            for (w, x), c in cwt.items():
                if c < cfg.as_min:
                    continue
                lift = (c / cw[w]) / (ct[x] / D)
                if lift >= cfg.as_lift:
                    pairs[w].append((x, math.log(lift)))
            t.assoc = {w: sorted(v, key=lambda e: (-e[1], e[0])) for w, v in pairs.items()}
        t.cc1 = {k: (v[0] + 0.2) / (v[1] + 2.0) for k, v in c1.items()}
        t.cc2 = {k: (v[0] + 0.2) / (v[1] + 2.0) for k, v in c2.items() if v[1] >= cfg.cc2_min}
        return t

    def set_tokenizer_flags(self, tokenizer):
        """c_tick / c_nl need the token text: (bytes of token t contain '`' / '\\n')."""
        self._tk = tokenizer
        self._flag_cache = {}

    def token_flag_table(self, vocab=151936):
        out = bytearray(vocab)
        for t in range(min(vocab, 151643)):
            n, nl = self.tok_flag(t)
            out[t] = min(n, 7) | (8 if nl else 0)
        return bytes(out)

    def tok_flag(self, t):
        """(number of '`' bytes, contains a '\n' byte) of token t's text; (0, False) for
        special tokens (ids >= 151643)."""
        tf = getattr(self, "tflg", None)
        if tf is not None:
            v = tf[t] if 0 <= t < len(tf) else 0
            return (v & 7, bool(v & 8))
        f = self._flag_cache.get(t)
        if f is None:
            if t >= 151643:
                f = (0, False)
            else:
                s = self._tk.decode([t])
                f = (s.count("`"), "\n" in s)
            self._flag_cache[t] = f
        return f

    # -- serialization --
    def save(self, path):
        cfg = self.cfg
        secs = []
        keys = []
        for ctx, (tot, top) in self.ng.items():
            keys.append((ctx_key(ctx), tot, top))
        keys.sort(key=lambda x: x[0])
        for a, b in zip(keys, keys[1:]):
            assert a[0] != b[0], "n-gram key collision"
        ngky = struct.pack(f"<{len(keys)}Q", *[k[0] for k in keys])
        vl = bytearray()
        nx = bytearray()
        first = 0
        for _, tot, top in keys:
            assert len(top) <= 7
            vl += struct.pack("<II", tot, (first << 3) | len(top))
            for tok, c in top:
                nx += struct.pack("<II", tok, c)
            first += len(top)
        secs += [(b"NGKY", len(keys), ngky), (b"NGVL", len(keys), bytes(vl)), (b"NGNX", first, bytes(nx))]
        words = sorted(((fnv1a64(w), w) for w in self.vocab), key=lambda x: x[0])
        for a, b in zip(words, words[1:]):
            assert a[0] != b[0], "word key collision"
        wk = struct.pack(f"<{len(words)}Q", *[h for h, _ in words])
        wv = bytearray()
        po = []
        for h, w in words:
            idf, plist = self.vocab[w]
            wv += struct.pack("<fII", idf, len(po), len(plist))
            po += plist
        secs += [(b"WDKY", len(words), wk), (b"WDVL", len(words), bytes(wv)),
                 (b"POST", len(po), struct.pack(f"<{len(po)}I", *po)),
                 (b"DNRM", len(self.norms), struct.pack(f"<{len(self.norms)}f", *self.norms))]
        offs = [0]
        toks = []
        for s in self.seqs:
            toks += s
            offs.append(len(toks))
        secs += [(b"SQOF", len(offs), struct.pack(f"<{len(offs)}I", *offs)),
                 (b"SQTK", len(toks), struct.pack(f"<{len(toks)}I", *toks))]
        cp = sorted(self.prop.items())
        secs.append((b"CPTK", len(cp), b"".join(struct.pack("<If", k, v) for k, v in cp)))
        c1 = sorted(self.cc1.items())
        secs.append((b"CPC1", len(c1), b"".join(struct.pack("<If", k, v) for k, v in c1)))
        c2 = sorted(((a << 32) | b, v) for (a, b), v in self.cc2.items())
        secs.append((b"CPC2", len(c2), b"".join(struct.pack("<QfI", k, v, 0) for k, v in c2)))
        if cfg.flags & FLAG_ASSOC:
            aw = sorted(((fnv1a64(w), w) for w in self.assoc), key=lambda x: x[0])
            for x, y2 in zip(aw, aw[1:]):
                assert x[0] != y2[0], "assoc word key collision"
            av = bytearray()
            at = bytearray()
            first = 0
            for h, w in aw:
                lst = self.assoc[w]
                av += struct.pack("<II", first, len(lst))
                for tok, ll in lst:
                    at += struct.pack("<If", tok, ll)
                first += len(lst)
            secs += [(b"ASWK", len(aw), struct.pack(f"<{len(aw)}Q", *[h for h, _ in aw])),
                     (b"ASWV", len(aw), bytes(av)), (b"ASTK", first, bytes(at))]
        if cfg.flags & FLAG_EDIT:
            po2 = [0]
            pb = bytearray()
            for txt in self.ptexts:
                pb += txt.encode("utf-8")
                po2.append(len(pb))
            secs += [(b"PTXO", len(po2), struct.pack(f"<{len(po2)}I", *po2)), (b"PTXB", len(pb), bytes(pb))]
        tf = self.token_flag_table(151936)
        nz = [(i, v) for i, v in enumerate(tf) if v]
        secs.append((b"TFLS", len(nz), b"".join(struct.pack("<II", i, v) for i, v in nz)))
        W = self.W
        flat = []
        for row in W["W1"]:
            flat += row
        flat += W["b1"] + W["W2"] + [W["b2"]] + W["WL"]
        secs.append((b"MLP0", len(flat), struct.pack(f"<{len(flat)}f", *flat)))
        hdr_len = 64 + 16 * len(secs)
        body = bytearray()
        dirs = bytearray()
        off = (hdr_len + 7) // 8 * 8
        for tag, cnt, data in secs:
            dirs += struct.pack("<4sIQ", tag, cnt, off)
            pad = (-len(data)) % 8
            body += data + b"\0" * pad
            off += len(data) + pad
        hdr = struct.pack("<8sII" + "I" * 8 + "IIII", b"ILDRAFT1", 1, len(secs), cfg.N, cfg.M, cfg.KNB, cfg.MAXM,
                          NF, cfg.H, cfg.minc_hi, len(self.seqs), cfg.flags, cfg.as_top, cfg.edit_E, 0)
        assert len(hdr) == 64
        blob = hdr + dirs
        blob += b"\0" * ((-len(blob)) % 8)
        blob += body
        with open(path, "wb") as f:
            f.write(blob)
        return len(blob)

    @staticmethod
    def load(path):
        b = open(path, "rb").read()
        magic, ver, nsec, N, M, KNB, MAXM, F, H, minc, D, flags, as_top, edit_E, _ = struct.unpack_from(
            "<8sII" + "I" * 8 + "IIII", b, 0)
        assert magic == b"ILDRAFT1" and ver == 1, (magic, ver)
        assert F == NF, (F, NF)
        cfg = Config(N=N, M=M, KNB=KNB, MAXM=MAXM, H=H, minc_hi=minc, flags=flags, as_top=as_top, edit_E=edit_E)
        t = Tables(cfg)
        sec = {}
        for i in range(nsec):
            tag, cnt, off = struct.unpack_from("<4sIQ", b, 64 + 16 * i)
            sec[tag] = (cnt, off)

        def arr(tag, fmt, size):
            cnt, off = sec[tag]
            return [struct.unpack_from(fmt, b, off + size * i) for i in range(cnt)]
        keys = [k[0] for k in arr(b"NGKY", "<Q", 8)]
        vals = arr(b"NGVL", "<II", 8)
        nx = arr(b"NGNX", "<II", 8)
        t.ngk = {k: (tot, [tuple(nx[j]) for j in range((fn >> 3), (fn >> 3) + (fn & 7))]) for k, (tot, fn) in zip(keys, vals)}
        wk = [k[0] for k in arr(b"WDKY", "<Q", 8)]
        wv = arr(b"WDVL", "<fII", 12)
        po = [p[0] for p in arr(b"POST", "<I", 4)]
        t.vocabk = {h: (idf, po[first:first + n]) for h, (idf, first, n) in zip(wk, wv)}
        t.norms = [x[0] for x in arr(b"DNRM", "<f", 4)]
        offs = [x[0] for x in arr(b"SQOF", "<I", 4)]
        toks = [x[0] for x in arr(b"SQTK", "<I", 4)]
        t.seqs = [toks[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        t.prop = {k: v for k, v in arr(b"CPTK", "<If", 8)}
        t.cc1 = {k: v for k, v in arr(b"CPC1", "<If", 8)}
        t.cc2 = {(k >> 32, k & 0xffffffff): v for k, v, _ in arr(b"CPC2", "<QfI", 16)}
        if flags & FLAG_ASSOC:
            awk = [k[0] for k in arr(b"ASWK", "<Q", 8)]
            awv = arr(b"ASWV", "<II", 8)
            atk = arr(b"ASTK", "<If", 8)
            t.assock = {h: [tuple(atk[j]) for j in range(first, first + n)] for h, (first, n) in zip(awk, awv)}
        if flags & FLAG_EDIT:
            po2 = [x[0] for x in arr(b"PTXO", "<I", 4)]
            cnt, off = sec[b"PTXB"]
            pb = b[off:off + cnt]
            t.ptexts = [pb[po2[i]:po2[i + 1]].decode("utf-8") for i in range(len(po2) - 1)]
        if b"TFLG" in sec:
            cnt, off = sec[b"TFLG"]
            t.tflg = b[off:off + cnt]
        if b"TFLS" in sec:
            tfl = bytearray(151936)
            for i, v in arr(b"TFLS", "<II", 8):
                tfl[i] = v
            t.tflg = bytes(tfl)
        flat = [x[0] for x in arr(b"MLP0", "<f", 4)]
        W1 = [flat[h * F:(h + 1) * F] for h in range(H)]
        p = H * F
        b1 = flat[p:p + H]
        p += H
        W2 = flat[p:p + H]
        p += H
        b2 = flat[p]
        p += 1
        WL = flat[p:p + F]
        t.W = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "WL": WL}
        t.loaded = True
        return t

    # unified lookups (work for built and loaded tables)
    def ng_get(self, ctx_tuple):
        if getattr(self, "loaded", False):
            return self.ngk.get(ctx_key(ctx_tuple))
        return self.ng.get(ctx_tuple)

    def assoc_get(self, w):
        if getattr(self, "loaded", False):
            return self.assock.get(fnv1a64(w)) if hasattr(self, "assock") else None
        return self.assoc.get(w)

    def vocab_get(self, w):
        if getattr(self, "loaded", False):
            return self.vocabk.get(fnv1a64(w))
        return self.vocab.get(w)

    def np_weights(self):
        import numpy as np
        W = self.W
        return (np.array(W["W1"], dtype=np.float64), np.array(W["b1"], dtype=np.float64),
                np.array(W["W2"], dtype=np.float64), float(W["b2"]), np.array(W["WL"], dtype=np.float64))


# ------------------------------------------------------------------------------------------
# drafter

class Drafter:
    def __init__(self, tables: Tables, tokenizer=None, weights=None):
        self.t = tables
        self.cfg = tables.cfg
        self.tok = tokenizer
        if tokenizer is not None and not hasattr(tables, "_tk"):
            tables.set_tokenizer_flags(tokenizer)
        self.w = weights if weights is not None else (tables.np_weights() if tables.W is not None else None)

    # --- per request ---
    def begin(self, pids, text):
        cfg, t = self.cfg, self.t
        self.P = list(pids)
        self.Pset = set(self.P)
        # neighbours: TF-IDF cosine, top KNB by (score desc, doc asc)
        sc = defaultdict(float)
        q2 = 0.0
        for w in prompt_words(text):
            e = t.vocab_get(w)
            if e is None:
                continue
            idf, plist = e
            q2 += idf * idf
            for j in plist:
                sc[j] += idf * idf
        qn = math.sqrt(q2) or 1.0
        ranked = sorted(((s / (qn * t.norms[j]), j) for j, s in sc.items()), key=lambda x: (-x[0], x[1]))
        self.nb = []
        for s, j in ranked[:cfg.KNB]:
            seq = TAIL + t.seqs[j]
            idx = {}
            L = len(seq)
            for n in range(1, cfg.MAXM + 1):
                for i in range(0, L - n):
                    k = tuple(seq[i:i + n])
                    if k not in idx:
                        idx[k] = i + n
            self.nb.append((s, seq, idx))
        # edited neighbours: the top edit_E neighbours' outputs with the prompts' differing words swapped
        self.ed = []
        if (cfg.flags & FLAG_EDIT) and self.tok is not None:
            from edit import edited_sequence
            for s, j in ranked[:min(cfg.edit_E, cfg.KNB)]:
                es = edited_sequence(self.tok, text, t.ptexts[j], t.seqs[j])
                if es is None:
                    continue
                seq = TAIL + es
                idx = {}
                L = len(seq)
                for n in range(1, cfg.MAXM + 1):
                    for i in range(0, L - n):
                        k = tuple(seq[i:i + n])
                        if k not in idx:
                            idx[k] = i + n
                self.ed.append((s, seq, idx))
        # space variants of prompt words
        self.spacevar = set()
        if self.tok is not None:
            for m in _SPACEVAR_RE.finditer(text):
                ids = self.tok.encode(" " + m.group(0), add_special_tokens=False).ids
                if ids and ids[0] not in self.Pset:
                    self.spacevar.add(ids[0])
        self.copy_cands = sorted(self.Pset | self.spacevar)
        # association: token -> (max loglift over prompt words, number of prompt words associating it)
        self.as_map = {}
        self.as_cands = []
        if cfg.flags & FLAG_ASSOC:
            for w in prompt_words(text):
                lst = t.assoc_get(w)
                if not lst:
                    continue
                for rank, (tok, ll) in enumerate(lst):
                    e = self.as_map.get(tok)
                    self.as_map[tok] = (max(e[0], ll), e[1] + 1) if e else (ll, 1)
                    if rank < cfg.as_top and (cfg.flags & FLAG_ASSOC_CANDS):
                        self.as_cands.append(tok)
            self.as_cands = sorted(set(self.as_cands))

    # --- candidates and features for the next token after ctx = TAIL + out ---
    def features(self, out):
        cfg, t = self.cfg, self.t
        ctx = TAIL + out
        L = len(ctx)
        cand = {}

        def row(tok):
            r = cand.get(tok)
            if r is None:
                r = [0.0] * NF
                r[0] = 1.0
                cand[tok] = r
            return r

        # n-gram
        dists = {}
        for n in range(min(cfg.N, L), 0, -1):
            e = t.ng_get(tuple(ctx[L - n:]))
            if e is not None:
                dists[n] = e
        long_n = max(dists) if dists else 0
        supp3 = max((n for n, e in dists.items() if e[0] >= 3), default=0)
        supp10 = max((n for n, e in dists.items() if e[0] >= 10), default=0)
        for n, (tot, top) in dists.items():
            for rk, (tok, c) in enumerate(top):
                r = row(tok)
                if n == long_n:
                    if rk == 1:
                        r[FI["ng_rank2"]] = 1.0
                    elif rk >= 2:
                        r[FI["ng_rank3plus"]] = 1.0
        # self lookup: most recent earlier occurrence of the longest suffix of out
        Lo = len(out)
        for n in range(min(cfg.MAXM, Lo - 1), 0, -1):
            pat = out[Lo - n:]
            hit = -1
            for j in range(Lo - n - 1, -1, -1):
                if out[j:j + n] == pat:
                    hit = j + n
                    break
            if hit >= 0:
                r = row(out[hit])
                r[FI["sf_m"]] = n / 4.0
                r[FI["sf_any"]] = 1.0
                r[FI["sf_m_ge3"]] = 1.0 if n >= 3 else 0.0
                break
        # prompt lookup: all occurrences of the longest suffix of out in P
        P = self.P
        for n in range(min(cfg.MAXM, Lo), 0, -1):
            pat = out[Lo - n:]
            occ = [j for j in range(0, len(P) - n) if P[j:j + n] == pat]
            if occ:
                for oi, j in enumerate(occ):
                    r = row(P[j + n])
                    r[FI["pr_m"]] = n / 4.0
                    r[FI["pr_any"]] = 1.0
                    r[FI["pr_m_ge2"]] = 1.0 if n >= 2 else 0.0
                    if oi == 0:
                        r[FI["pr_first"]] = 1.0
                break
        # neighbours
        votes = defaultdict(list)
        for rank, (s, seq, idx) in enumerate(self.nb):
            for n in range(min(cfg.MAXM, L), 0, -1):
                j = idx.get(tuple(ctx[L - n:]))
                if j is not None:
                    votes[seq[j]].append((s, n, rank == 0))
                    break
        nnb = max(1, len(self.nb))
        for tok, vs in votes.items():
            r = row(tok)
            r[FI["nb_sum"]] = sum(s for s, _, _ in vs)
            r[FI["nb_msum"]] = sum(s * min(n, 8) / 8.0 for s, n, _ in vs)
            r[FI["nb_cnt"]] = len(vs) / nnb
            r[FI["nb_maxm"]] = max(min(n, 8) for _, n, _ in vs) / 8.0
            for s, n, first in vs:
                if first:
                    r[FI["nb_top1"]] = 1.0
                    r[FI["nb_top1_m"]] = min(n, 8) / 8.0
        # edited neighbours
        evotes = defaultdict(list)
        for rank, (s, seq, idx) in enumerate(self.ed):
            for n in range(min(cfg.MAXM, L), 0, -1):
                j = idx.get(tuple(ctx[L - n:]))
                if j is not None:
                    evotes[seq[j]].append((s, n, rank == 0))
                    break
        ned = max(1, len(self.ed))
        for tok, vs in evotes.items():
            r = row(tok)
            r[FI["ed_sum"]] = sum(s for s, _, _ in vs)
            r[FI["ed_cnt"]] = len(vs) / ned
            r[FI["ed_maxm"]] = max(min(n, 8) for _, n, _ in vs) / 8.0
            for s, n, first in vs:
                if first:
                    r[FI["ed_top1"]] = 1.0
        # copy candidates
        used = set(out)
        c1 = t.cc1.get(ctx[-1], 0.1)
        c2 = t.cc2.get((ctx[-2], ctx[-1]), 0.1)
        first_unused = -1
        for x in P:
            if x not in used and t.prop.get(x, 0.2) > 0.3:
                first_unused = x
                break
        for x in self.copy_cands:
            r = row(x)
            inp = x in self.Pset
            pt = t.prop.get(x, 0.2) if inp else 0.3
            r[FI["cp_prop"]] = math.log(pt / (1.0 - pt)) / 4.0
            r[FI["cp_ctx1"]] = c1
            r[FI["cp_ctx2"]] = c2
            r[FI["cp_unused"]] = 0.0 if x in used else 1.0
            r[FI["cp_first_unused"]] = 1.0 if x == first_unused else 0.0
            r[FI["cp_spacevar"]] = 0.0 if inp else 1.0
        for x in self.as_cands:
            row(x)[FI["as_cand"]] = 1.0
        # per-candidate n-gram probabilities and token features
        open_fence = 1.0 if out and out[0] in FENCE_OPEN else 0.0
        open_tick = 1.0 if out and out[0] == TICK else 0.0
        nticks = 0
        for x in out:
            nticks += t.tok_flag(x)[0]
        ticks_odd = float(nticks % 2)
        st_len = min(Lo, 32) / 32.0
        top_long = None
        ent_low = False
        if long_n:
            tot, top = dists[long_n]
            top_long = top[0][0]
            ent = -sum((c / tot) * math.log(c / tot) for _, c in top)
            ent_low = ent < 0.3
        for tok, r in cand.items():
            for n, (tot, top) in dists.items():
                c = 0
                for tk, cc in top:
                    if tk == tok:
                        c = cc
                        break
                if c:
                    p = c / tot
                    r[FI["ng_p%d" % n]] = p if n <= 6 else 0.0
                    if n == long_n:
                        r[FI["ng_p_long"]] = p
                        r[FI["ng_ord_long"]] = n / 6.0 * p
                        r[FI["ng_lcnt_long"]] = math.log1p(c)
                    if n == supp3:
                        r[FI["ng_p_supp3"]] = p
                    if n == supp10:
                        r[FI["ng_p_supp10"]] = p
            if tok == top_long:
                r[FI["ng_top_long"]] = 1.0
                if ent_low:
                    r[FI["ng_ent_low"]] = 1.0
            r[FI["is_stop"]] = 1.0 if tok in STOPS else 0.0
            r[FI["in_prompt"]] = 1.0 if tok in self.Pset else 0.0
            r[FI["agree_ng_nb"]] = r[FI["ng_top_long"]] * min(1.0, 2.0 * r[FI["nb_cnt"]])
            r[FI["agree_sf_nb"]] = r[FI["sf_any"]] * min(1.0, 2.0 * r[FI["nb_cnt"]])
            ft = t.tok_flag(tok)
            r[FI["st_open_fence"]] = open_fence
            r[FI["st_open_tick"]] = open_tick
            r[FI["c_tick"]] = 1.0 if ft[0] > 0 else 0.0
            r[FI["c_nl"]] = 1.0 if ft[1] else 0.0
            r[FI["st_ticks_odd"]] = ticks_odd
            r[FI["st_len"]] = st_len
            r[FI["c_repeat"]] = 1.0 if tok in used else 0.0
            ae = self.as_map.get(tok)
            if ae is not None:
                r[FI["as_max"]] = ae[0] / 4.0
                r[FI["as_cnt"]] = min(ae[1], 4) / 4.0
        toks = sorted(cand)
        return toks, [cand[x] for x in toks]

    # --- scoring ---
    def probs(self, out):
        """Returns [(token, prob)] sorted by prob desc, token asc (null option excluded)."""
        import numpy as np
        toks, rows = self.features(out)
        if not toks:
            return []
        X = np.array(rows, dtype=np.float64)
        W1, b1, W2, b2, WL = self.w
        s = np.maximum(X @ W1.T + b1, 0.0) @ W2 + b2 + X @ WL
        m = max(float(s.max()), 0.0)
        e = np.exp(s - m)
        Z = float(e.sum()) + math.exp(-m)
        ps = e / Z
        order = sorted(range(len(toks)), key=lambda i: (-ps[i], toks[i]))
        return [(toks[i], float(ps[i])) for i in order]

    def chain(self, out, K, tau=0.0):
        o = list(out)
        d = []
        while len(d) < K:
            pr = self.probs(o)
            if not pr or pr[0][1] < tau:
                break
            tok = pr[0][0]
            d.append(tok)
            o.append(tok)
            if tok in STOPS:
                break
        return d

    def tree(self, out, B, branch=8):
        """Best-first draft tree with B nodes. Returns (parents, tokens): node i has parent
        parents[i] (-1 = the root = last accepted token) and token tokens[i]. Nodes are listed in
        expansion order, so parents[i] < i."""
        parents, tokens = [], []
        heap = []
        tie = 0
        ctxs = {-1: list(out)}
        for tok, p in self.probs(out)[:branch]:
            tie += 1
            heapq.heappush(heap, (-p, tie, -1, tok))
        while heap and len(tokens) < B:
            negpp, _, par, tok = heapq.heappop(heap)
            nid = len(tokens)
            parents.append(par)
            tokens.append(tok)
            o = ctxs[par] + [tok]
            ctxs[nid] = o
            if tok in STOPS or len(tokens) >= B:
                continue
            for tok2, p2 in self.probs(o)[:branch]:
                tie += 1
                heapq.heappush(heap, (negpp * p2, tie, nid, tok2))
        return parents, tokens


def accept_tree(parents, tokens, y, p):
    """Longest path from the root matching y[p], y[p+1], ...; returns its length."""
    children = defaultdict(dict)
    for i, (par, tok) in enumerate(zip(parents, tokens)):
        children[par].setdefault(tok, i)
    node, a = -1, 0
    while p + a < len(y) and y[p + a] in children[node]:
        node = children[node][y[p + a]]
        a += 1
    return a
