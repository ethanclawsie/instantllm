#!/usr/bin/env python3
"""Write a GGUF holding exactly the models/hf weights, for like-for-like llama.cpp checks.

Why: every GGUF in Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF (models/gguf/, including "fp16")
was converted from the Instruct-AWQ checkpoint (general.name "Qwen2.5 Coder 0.5B Instruct
AWQ"): the AWQ scales are folded into the RMSNorm weights and the linear weights are
dequantized 4-bit values, so they are not the HF bf16 weights.

This copies every metadata key/value (tokenizer, hyper-parameters) byte for byte from
the official fp16 GGUF, except general.name / general.finetune / general.file_type, and
writes the HF tensors under llama.cpp's qwen2 names (no permutation; tied embeddings,
so no output.weight). Matrices are F16 (exact for these bf16 values except ones below
the f16 normal range) or F32 with --f32; norms and biases are F32, as in the original.

  .venv/bin/python -I tools/reference/make_gguf.py            # scratch/reference/gguf/qwen2.5-coder-0.5b-instruct-hf-f16.gguf
  .venv/bin/python -I tools/reference/make_gguf.py --f32
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcommon as rc  # noqa: E402
from qwen_ref import N_LAYERS, load_safetensors_bf16  # noqa: E402

ALIGN = 32
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_kv_raw(path: Path) -> list[tuple[str, bytes]]:
    """Return [(key, raw bytes of the whole key/value record)] of a GGUF v3 file."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def skip_str():
            (n,) = rd("Q")
            return f.read(n)

        def skip_val(t):
            if t in SCALAR:
                rd(SCALAR[t])
            elif t == 8:
                skip_str()
            elif t == 9:
                (et,) = rd("I")
                (n,) = rd("Q")
                for _ in range(n):
                    skip_val(et)
            else:
                raise ValueError(f"unknown GGUF value type {t}")

        assert f.read(4) == b"GGUF"
        (ver,) = rd("I")
        assert ver == 3, ver
        rd("Q")  # n_tensors
        (nkv,) = rd("Q")
        out = []
        for _ in range(nkv):
            start = f.tell()
            key = skip_str().decode()
            (t,) = rd("I")
            skip_val(t)
            end = f.tell()
            f.seek(start)
            out.append((key, f.read(end - start)))
        return out


def gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(key: str, val: str) -> bytes:
    return gstr(key) + struct.pack("<I", 8) + gstr(val)


def kv_u32(key: str, val: int) -> bytes:
    return gstr(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-gguf", type=Path, default=rc.REPO / "models/gguf/qwen2.5-coder-0.5b-instruct-fp16.gguf",
                    help="GGUF whose metadata (tokenizer, hyper-parameters) is copied")
    ap.add_argument("--f32", action="store_true", help="store matrices as F32 instead of F16")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    kind = "f32" if args.f32 else "f16"
    out = args.out or rc.SCRATCH_DIR / "gguf" / f"qwen2.5-coder-0.5b-instruct-hf-{kind}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)

    W = load_safetensors_bf16(rc.SAFETENSORS)
    names = [("token_embd.weight", "model.embed_tokens.weight"), ("output_norm.weight", "model.norm.weight")]
    for i in range(N_LAYERS):
        p, b = f"blk.{i}.", f"model.layers.{i}."
        names += [
            (p + "attn_norm.weight", b + "input_layernorm.weight"),
            (p + "attn_q.weight", b + "self_attn.q_proj.weight"), (p + "attn_q.bias", b + "self_attn.q_proj.bias"),
            (p + "attn_k.weight", b + "self_attn.k_proj.weight"), (p + "attn_k.bias", b + "self_attn.k_proj.bias"),
            (p + "attn_v.weight", b + "self_attn.v_proj.weight"), (p + "attn_v.bias", b + "self_attn.v_proj.bias"),
            (p + "attn_output.weight", b + "self_attn.o_proj.weight"),
            (p + "ffn_norm.weight", b + "post_attention_layernorm.weight"),
            (p + "ffn_gate.weight", b + "mlp.gate_proj.weight"),
            (p + "ffn_up.weight", b + "mlp.up_proj.weight"),
            (p + "ffn_down.weight", b + "mlp.down_proj.weight"),
        ]
    assert len(names) == len(W), (len(names), len(W))

    tensors = []  # (name, ggml type, array)
    for gname, hname in names:
        a = W[hname]
        if a.ndim == 2 and not args.f32:
            tensors.append((gname, 1, a.astype("<f2")))
        else:
            tensors.append((gname, 0, a.astype("<f4")))

    replace = {
        "general.name": kv_str("general.name", f"Qwen2.5 Coder 0.5B Instruct (HF bf16 -> {kind})"),
        "general.finetune": kv_str("general.finetune", "Instruct"),
        "general.file_type": kv_u32("general.file_type", 0 if args.f32 else 1),
    }
    kvs = [replace.get(k, raw) for k, raw in read_kv_raw(args.src_gguf)]

    infos, offset = [], 0
    for gname, gtype, a in tensors:
        dims = list(reversed(a.shape))  # ggml ne0 is the fastest-varying dimension
        infos.append(gstr(gname) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
                     + struct.pack("<IQ", gtype, offset))
        offset += (a.nbytes + ALIGN - 1) // ALIGN * ALIGN

    with open(out, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        for kv in kvs:
            f.write(kv)
        for info in infos:
            f.write(info)
        f.write(b"\0" * (-f.tell() % ALIGN))
        for _, _, a in tensors:
            f.write(a.tobytes())
            f.write(b"\0" * (-a.nbytes % ALIGN))
    sub = sum(int(np.count_nonzero((np.abs(W[h]) < 6.104e-05) & (W[h] != 0))) for _, h in names if W[h].ndim == 2)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(tensors)} tensors"
          + ("" if args.f32 else f"; {sub} nonzero matrix weights below the f16 normal range") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
