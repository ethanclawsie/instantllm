#!/usr/bin/env python3
"""Numpy reference forward pass for Qwen2.5-Coder-0.5B-Instruct, and golden-data generator.

A deliberately plain implementation: every tensor is float32 (or float64 with --dtype f64),
bf16 weights are widened exactly by bit shifting, prompts are processed in one batched
prefill, then tokens are decoded one at a time against a KV cache with greedy argmax.

Model (Qwen2ForCausalLM): 24 layers, hidden 896, SwiGLU MLP 4864, 14 query heads and 2 KV
heads of dim 64 (GQA: query head h reads KV head h // 7), q/k/v biases, no o_proj bias,
RoPE theta 1e6 in rotate_half form (pairs (i, i+32) of each head), RMSNorm eps 1e-6 with a
plain weight multiply, embeddings tied to the lm_head (all 151936 rows).

Usage (from the repo root):
  .venv/bin/python -I tools/reference/qwen_ref.py                  # regenerate golden/
  .venv/bin/python -I tools/reference/qwen_ref.py --prompts 0 --out /tmp/x --no-debug
  .venv/bin/python -I tools/reference/qwen_ref.py --dtype f64 --out scratch/reference/ref_f64 --no-debug
  .venv/bin/python -I tools/reference/qwen_ref.py --check-cache    # KV-cache consistency check

Runs on the CPU only (numpy + Accelerate BLAS), so it needs no GPU lock.
"""
from __future__ import annotations

import argparse
import json
import platform
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcommon as rc  # noqa: E402

N_LAYERS = 24
HIDDEN = 896
INTER = 4864
N_HEAD = 14
N_KV = 2
GROUP = N_HEAD // N_KV  # 7 query heads per KV head
HEAD_DIM = 64
HALF = HEAD_DIM // 2
VOCAB = rc.VOCAB
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
ATTN_SCALE = HEAD_DIM ** -0.5  # 0.125, exact in binary floating point


def load_safetensors_bf16(path: Path) -> dict[str, np.ndarray]:
    """Parse a safetensors file whose tensors are all BF16; return float32 arrays.

    bf16 -> f32 is exact: the bf16 bits are the high 16 bits of the f32 encoding.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
        a, b = info["data_offsets"]
        count = int(np.prod(info["shape"]))
        assert b - a == 2 * count, name
        u16 = np.frombuffer(data, dtype="<u2", count=count, offset=a)
        out[name] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    return out


def rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + RMS_EPS))) * w


def rotate_half(x: np.ndarray) -> np.ndarray:
    return np.concatenate([-x[..., HALF:], x[..., :HALF]], axis=-1)


class KVCache:
    def __init__(self, max_len: int, dtype):
        self.k = [np.zeros((N_KV, max_len, HEAD_DIM), dtype) for _ in range(N_LAYERS)]
        self.v = [np.zeros((N_KV, max_len, HEAD_DIM), dtype) for _ in range(N_LAYERS)]
        self.len = 0


class Qwen2Ref:
    def __init__(self, path: Path = rc.SAFETENSORS, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        W = load_safetensors_bf16(path)

        def take(name):
            return np.ascontiguousarray(W.pop(name), dtype=self.dtype)

        self.embed = take("model.embed_tokens.weight")  # [V, H]; also the lm_head
        self.norm = take("model.norm.weight")
        self.layers = []
        for i in range(N_LAYERS):
            p = f"model.layers.{i}."
            self.layers.append(SimpleNamespace(
                ln1=take(p + "input_layernorm.weight"),
                ln2=take(p + "post_attention_layernorm.weight"),
                wq=take(p + "self_attn.q_proj.weight"), bq=take(p + "self_attn.q_proj.bias"),
                wk=take(p + "self_attn.k_proj.weight"), bk=take(p + "self_attn.k_proj.bias"),
                wv=take(p + "self_attn.v_proj.weight"), bv=take(p + "self_attn.v_proj.bias"),
                wo=take(p + "self_attn.o_proj.weight"),
                wg=take(p + "mlp.gate_proj.weight"),
                wu=take(p + "mlp.up_proj.weight"),
                wd=take(p + "mlp.down_proj.weight"),
            ))
        if W:
            raise ValueError(f"unused tensors: {sorted(W)[:5]}")
        assert self.embed.shape == (VOCAB, HIDDEN)
        # RoPE frequencies and angles in float64, cos/sin rounded once to the compute dtype.
        self.inv_freq = 1.0 / ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)

    def new_cache(self, max_len: int) -> KVCache:
        return KVCache(max_len, self.dtype)

    def rope_tables(self, pos0: int, n: int):
        ang = np.arange(pos0, pos0 + n, dtype=np.float64)[:, None] * self.inv_freq[None, :]
        ang = np.concatenate([ang, ang], axis=-1)  # [n, 64]: column j and j+32 share a frequency
        return np.cos(ang).astype(self.dtype), np.sin(ang).astype(self.dtype)

    def forward(self, ids, cache: KVCache, all_logits: bool = False, dbg: dict | None = None):
        """Run len(ids) new tokens at positions cache.len .. cache.len+n-1.

        Returns logits [n, V] if all_logits else [V] for the last new token. With dbg, the
        per-layer debug tensors described in README.md are stored into the dict.
        """
        ids = np.asarray(ids, dtype=np.int64)
        n = ids.shape[0]
        pos0 = cache.len
        total = pos0 + n
        cos, sin = self.rope_tables(pos0, n)
        cos, sin = cos[:, None, :], sin[:, None, :]
        # causal mask: new token t (absolute position pos0+t) sees cache positions <= pos0+t
        masked = np.arange(total)[None, :] > (pos0 + np.arange(n))[:, None]  # [n, total]

        x = self.embed[ids]  # [n, H]
        if dbg is not None:
            dbg["resid"] = [x[-1].copy()]
        for li, L in enumerate(self.layers):
            # --- attention ---
            h = rmsnorm(x, L.ln1)
            q = (h @ L.wq.T + L.bq).reshape(n, N_HEAD, HEAD_DIM)
            k = (h @ L.wk.T + L.bk).reshape(n, N_KV, HEAD_DIM)
            v = (h @ L.wv.T + L.bv).reshape(n, N_KV, HEAD_DIM)
            q_pre, k_pre = q, k
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            cache.k[li][:, pos0:total] = k.transpose(1, 0, 2)
            cache.v[li][:, pos0:total] = v.transpose(1, 0, 2)
            K = cache.k[li][:, :total]  # [kv, total, 64]
            V = cache.v[li][:, :total]
            qg = q.transpose(1, 0, 2).reshape(N_KV, GROUP, n, HEAD_DIM)  # head h -> (h//7, h%7)
            s = np.matmul(qg, K[:, None].transpose(0, 1, 3, 2)) * ATTN_SCALE  # [kv, g, n, total]
            s = np.where(masked, -np.inf, s)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(axis=-1, keepdims=True)
            o = np.matmul(p, V[:, None])  # [kv, g, n, 64]
            o = o.reshape(N_HEAD, n, HEAD_DIM).transpose(1, 0, 2).reshape(n, HIDDEN)
            attn_out = o @ L.wo.T
            x_mid = x + attn_out
            # --- MLP ---
            h2 = rmsnorm(x_mid, L.ln2)
            g = h2 @ L.wg.T
            u = h2 @ L.wu.T
            with np.errstate(over="ignore"):  # exp(-g) -> inf for g << 0 gives silu -> -0
                act = (g / (1.0 + np.exp(-g))) * u
            mlp_out = act @ L.wd.T
            x = x_mid + mlp_out
            if dbg is not None:
                dbg["resid"].append(x[-1].copy())
                if li == 0:
                    dbg.update(
                        l0_attn_norm=h, l0_q_pre_rope=q_pre, l0_k_pre_rope=k_pre,
                        l0_q=q, l0_k=k, l0_v=v, l0_attn=o, l0_attn_out=attn_out,
                        l0_resid_mid=x_mid, l0_mlp_norm=h2, l0_mlp_act=act,
                        l0_mlp_out=mlp_out, l0_resid_out=x,
                    )
        cache.len = total
        if not all_logits:
            x = x[-1:]
        xf = rmsnorm(x, self.norm)
        if dbg is not None:
            dbg["final_norm"] = xf[-1].copy()
        logits = xf @ self.embed.T  # tied lm_head
        return logits if all_logits else logits[0]

    def greedy(self, ids, max_new: int = rc.MAX_NEW, stop_ids=rc.STOP_IDS,
               keep_decode_logits: int = 0, dbg: dict | None = None) -> dict:
        """Greedy decoding. At most max_new tokens are sampled; a sampled stop token counts as
        one of them, ends generation and is not part of output_ids."""
        cache = self.new_cache(len(ids) + max_new + 1)
        logits = self.forward(ids, cache, dbg=dbg)
        last_logits = logits.copy()
        out, step_top5, decode_logits = [], [], []
        stop = None
        for _ in range(max_new):
            tok = int(np.argmax(logits))  # first maximum = lowest id on exact ties
            step_top5.append(rc.topk(logits, 5))
            if tok in stop_ids:
                stop = tok
                break
            out.append(tok)
            if len(out) == max_new:
                break
            logits = self.forward([tok], cache)
            if len(decode_logits) < keep_decode_logits:
                decode_logits.append(logits.copy())
        return dict(output_ids=out, stop_token=stop,
                    finish_reason="stop" if stop is not None else "length",
                    last_logits=last_logits, step_top5=step_top5,
                    decode_logits=decode_logits)


DEBUG_DESC = {
    "resid": "residual stream at the last prompt position: row 0 = token embedding (layer-0 input), row i+1 = after layer i",
    "final_norm": "model.norm output at the last prompt position (lm_head input)",
    "l0_attn_norm": "layer 0 input_layernorm output, all prompt positions",
    "l0_q_pre_rope": "layer 0 q_proj(x)+bias before RoPE, [pos, head, dim]",
    "l0_k_pre_rope": "layer 0 k_proj(x)+bias before RoPE, [pos, kv_head, dim]",
    "l0_q": "layer 0 q after bias and RoPE, [pos, head, dim]",
    "l0_k": "layer 0 k after bias and RoPE (what goes into the KV cache), [pos, kv_head, dim]",
    "l0_v": "layer 0 v after bias (no RoPE), [pos, kv_head, dim]",
    "l0_attn": "layer 0 attention context softmax(q k^T * 0.125 + causal mask) v, heads concatenated, before o_proj",
    "l0_attn_out": "layer 0 o_proj output (added to the residual)",
    "l0_resid_mid": "layer 0 residual after the attention add (embedding + attn_out)",
    "l0_mlp_norm": "layer 0 post_attention_layernorm output",
    "l0_mlp_act": "layer 0 silu(gate_proj(x)) * up_proj(x)",
    "l0_mlp_out": "layer 0 down_proj output (added to the residual)",
    "l0_resid_out": "layer 0 output residual, all prompt positions (last row == resid[1])",
    "decode_logits": "logits of decode step j: output_ids[j] fed at position n_input+j; argmax gives sampled token j+1",
}


def write_debug(out_dir: Path, i: int, rec: dict, dbg: dict, decode_logits: list) -> None:
    ddir = out_dir / f"debug_{rc.pname(i)}"
    ddir.mkdir(parents=True, exist_ok=True)
    tensors = dict(dbg)
    tensors["resid"] = np.stack(dbg["resid"])
    if decode_logits:
        tensors["decode_logits"] = np.stack(decode_logits)
    index = {
        "prompt_index": i,
        "n_input": rec["n_input"],
        "dtype": "float32 little-endian, C order (last dim fastest)",
        "decode_tokens": rec["output_ids"][: len(decode_logits)],
        "decode_positions": [rec["n_input"] + j for j in range(len(decode_logits))],
        "tensors": {},
    }
    for name in DEBUG_DESC:
        if name not in tensors:
            continue
        arr = np.asarray(tensors[name])
        rc.write_f32(ddir / f"{name}.bin", arr)
        index["tensors"][name] = {"file": f"{name}.bin", "shape": list(arr.shape), "desc": DEBUG_DESC[name]}
    rc.write_json(ddir / "index.json", index)


def check_cache(model: Qwen2Ref, ids, max_new: int = 8) -> None:
    """Compare KV-cached decode logits with a full prefill over the same token sequence."""
    r = model.greedy(ids, max_new=max_new + 1, keep_decode_logits=max_new)
    seq = list(ids)
    worst = 0.0
    for j, dl in enumerate(r["decode_logits"]):
        seq.append(r["output_ids"][j])
        full = model.forward(seq, model.new_cache(len(seq)))
        d = float(np.abs(full.astype(np.float64) - dl).max())
        worst = max(worst, d)
        print(f"  decode step {j}: max|cached - full prefill| = {d:.3e}  (argmax {int(full.argmax())} vs {int(dl.argmax())})")
    print(f"KV-cache consistency: worst max abs diff {worst:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=rc.GOLDEN_DIR, help="output directory (default: tools/reference/golden)")
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32", help="compute dtype (default f32)")
    ap.add_argument("--prompts", default="all", help="comma-separated prompt indices or 'all'")
    ap.add_argument("--max-new", type=int, default=rc.MAX_NEW)
    ap.add_argument("--debug-prompt", type=int, default=0, help="prompt index for per-layer dumps (-1: none)")
    ap.add_argument("--no-debug", action="store_true", help="skip the per-layer dumps")
    ap.add_argument("--n-decode-logits", type=int, default=8, help="decode-step logits kept for the debug prompt")
    ap.add_argument("--check-cache", action="store_true", help="only run the KV-cache consistency check on prompt 0")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "f32" else np.float64
    t0 = time.perf_counter()
    model = Qwen2Ref(rc.SAFETENSORS, dtype)
    print(f"loaded weights ({args.dtype}) in {time.perf_counter() - t0:.1f}s", flush=True)

    tok = rc.load_tokenizer()
    system = rc.load_system()
    prompts = rc.load_prompts()
    prefix = rc.prefix_text(system)
    prefix_ids = rc.encode(tok, prefix)

    if args.check_cache:
        check_cache(model, rc.encode(tok, rc.chat_text(system, prompts[0])))
        return 0

    sel = range(len(prompts)) if args.prompts == "all" else [int(s) for s in args.prompts.split(",")]
    debug_prompt = -1 if args.no_debug else args.debug_prompt
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for i in sel:
        text = rc.chat_text(system, prompts[i])
        ids = rc.encode(tok, text)
        if ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(f"prompt {i}: system prefix does not tokenize as a prefix")
        dbg = {} if i == debug_prompt else None
        t1 = time.perf_counter()
        r = model.greedy(ids, max_new=args.max_new,
                         keep_decode_logits=args.n_decode_logits if dbg is not None else 0, dbg=dbg)
        dt = time.perf_counter() - t1
        rec = {
            "index": i,
            "prompt": prompts[i],
            "input_ids": ids,
            "n_input": len(ids),
            "prefix_len": len(prefix_ids),
            "output_ids": r["output_ids"],
            "stop_token": r["stop_token"],
            "finish_reason": r["finish_reason"],
            "text": rc.decode(tok, r["output_ids"]),
            "top10": rc.topk(r["last_logits"], 10),
            "step_top5": r["step_top5"],
        }
        rc.write_record(out, i, rec, r["last_logits"])
        if dbg is not None:
            write_debug(out, i, rec, dbg, r["decode_logits"])
        index.append({k: rec[k] for k in ("index", "prompt", "n_input", "output_ids", "stop_token", "finish_reason", "text")})
        margin = rec["top10"][0][1] - rec["top10"][1][1]
        min_margin = min(s[0][1] - s[1][1] for s in r["step_top5"])
        print(f"[{i:2d}] in={len(ids):3d} out={len(r['output_ids']):2d} {dt:5.2f}s "
              f"margin0={margin:6.3f} min_margin={min_margin:6.3f}  {rec['text']!r}", flush=True)

    rc.write_json(out / "index.json", index)
    rc.write_json(out / "prefix.json", {
        "text": prefix,
        "ids": prefix_ids,
        "n": len(prefix_ids),
        "note": "token ids of the system-prompt prefix through '<|im_start|>user\\n'; "
                "equal to input_ids[:n] of every prompt",
    })
    import tokenizers
    rc.write_json(out / "meta.json", {
        "format_version": 1,
        "model": "Qwen2.5-Coder-0.5B-Instruct (Qwen2ForCausalLM)",
        "weights": "models/hf/model.safetensors",
        "weights_sha256": rc.sha256_file(rc.SAFETENSORS),
        "tokenizer": "models/hf/tokenizer.json",
        "compute_dtype": np.dtype(dtype).name,
        "config": {
            "layers": N_LAYERS, "hidden": HIDDEN, "intermediate": INTER, "heads": N_HEAD,
            "kv_heads": N_KV, "head_dim": HEAD_DIM, "vocab": VOCAB, "rms_eps": RMS_EPS,
            "rope_theta": ROPE_THETA, "rope": "rotate_half (non-interleaved)", "tied_embeddings": True,
        },
        "chat_template": rc.CHAT_TEMPLATE,
        "system_file": "bench/system.txt",
        "prompts_file": "bench/prompts.txt",
        "generation": {
            "decoding": "greedy argmax over all 151936 logits, exact ties -> lowest id",
            "max_new_tokens": args.max_new,
            "stop_ids": list(rc.STOP_IDS),
            "stop_token_counts_toward_max_new": True,
            "stop_token_in_output_ids": False,
        },
        "prompts_written": list(sel),
        "debug_prompt": debug_prompt,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "tokenizers": tokenizers.__version__},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"wrote {out} in {time.perf_counter() - t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
