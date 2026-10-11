"""Shared helpers for tools/reference: paths, chat template, tokenization and the
golden / engine-output file layout (documented in README.md).

Imported by qwen_ref.py, compare.py, run_mlx.py and run_llamacpp.py. Those scripts are
run with `python -I`, which does not put the script directory on sys.path, so each of
them inserts this directory explicitly before importing this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HF_DIR = REPO / "models" / "hf"
SAFETENSORS = HF_DIR / "model.safetensors"
TOKENIZER_JSON = HF_DIR / "tokenizer.json"
SYSTEM_TXT = REPO / "bench" / "system.txt"
PROMPTS_TXT = REPO / "bench" / "prompts.txt"
GOLDEN_DIR = HERE / "golden"
SCRATCH_DIR = REPO / "scratch" / "reference"  # gitignored: validation runs, builds

VOCAB = 151936          # embedding / lm_head rows (logits length)
STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>
MAX_NEW = 64            # at most 64 sampled tokens; a sampled stop token counts as one

PREFIX_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
CHAT_TEMPLATE = PREFIX_TEMPLATE + "{prompt}<|im_end|>\n<|im_start|>assistant\n"


def load_system() -> str:
    """Contents of bench/system.txt, used verbatim (the file has no trailing newline)."""
    return SYSTEM_TXT.read_text(encoding="utf-8")


def load_prompts() -> list[str]:
    """bench/prompts.txt, one prompt per line (trailing empty lines dropped)."""
    lines = PROMPTS_TXT.read_text(encoding="utf-8").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def chat_text(system: str, prompt: str) -> str:
    return CHAT_TEMPLATE.format(system=system, prompt=prompt)


def prefix_text(system: str) -> str:
    return PREFIX_TEMPLATE.format(system=system)


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_JSON))


def encode(tok, text: str) -> list[int]:
    # Special tokens such as <|im_start|> are matched as added tokens in the text. The
    # Qwen2 post-processor adds nothing (no BOS), so add_special_tokens is irrelevant;
    # False makes that explicit.
    return tok.encode(text, add_special_tokens=False).ids


def decode(tok, ids: list[int]) -> str:
    return tok.decode(list(ids), skip_special_tokens=False)


def pname(i: int) -> str:
    return f"p{i:02d}"


def write_f32(path: Path, arr) -> None:
    np.ascontiguousarray(np.asarray(arr), dtype="<f4").tofile(path)


def read_f32(path: Path, shape=None) -> np.ndarray:
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(shape) if shape is not None else a


def topk(logits: np.ndarray, k: int) -> list[list]:
    """Top-k as [[id, logit], ...], sorted by logit descending, ties by lower id."""
    logits = np.asarray(logits)
    idx = np.argpartition(-logits, k)[:k]
    # include every entry tied with the k-th value so tie-breaking by id is exact
    kth = logits[idx].min()
    idx = np.nonzero(logits >= kth)[0]
    order = np.lexsort((idx, -logits[idx].astype(np.float64)))
    idx = idx[order][:k]
    return [[int(i), float(logits[i])] for i in idx]


def greedy_seq(rec: dict) -> list[int]:
    """Sampled-token sequence of a record: output_ids plus the stop token, if any."""
    seq = list(rec.get("output_ids", []))
    if rec.get("stop_token") is not None:
        seq.append(int(rec["stop_token"]))
    return seq


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_record(out_dir: Path, i: int, rec: dict, logits=None) -> None:
    """Write pNN.json (and pNN.logits.bin when logits are given) in the shared layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if logits is not None:
        logits = np.asarray(logits).reshape(-1)
        assert logits.shape[0] == VOCAB, logits.shape
        write_f32(out_dir / f"{pname(i)}.logits.bin", logits)
        rec = dict(rec, last_logits_file=f"{pname(i)}.logits.bin")
    write_json(out_dir / f"{pname(i)}.json", rec)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
