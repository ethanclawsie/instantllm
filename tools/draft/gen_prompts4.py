"""Training-data expansion #3 (all TRAIN): more family expansions with extended slot values,
new compositional requests and new phrasing variants of TRAIN hand-written bases. Deduplicated
against every earlier prompt file; bench and held-out near-duplicates are rejected.

usage: .venv/bin/python -I tools/draft/gen_prompts4.py [--seed 4] [--n-comp 12000] [--per-fam 70]
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_prompts as G  # noqa: E402
import gen_prompts2 as G2  # noqa: E402
import gen_prompts3 as G3  # noqa: E402
from prompts_hand import HAND  # noqa: E402
from prompts_hand2 import HAND2  # noqa: E402

ROOT = G.ROOT
EXTRA = {
    "file": ["index.js", "app.py", "schema.sql", "config.toml", "go.mod", "Cargo.toml", "pom.xml", "notes.org",
             "draft.md", "main.rs", "server.ts", "build.gradle", "requirements.txt", "yarn.lock", ".bashrc"],
    "textfile": ["urls.txt", "emails.txt", "metrics.tsv", "events.log", "users.txt", "queue.txt", "scores.csv"],
    "logfile": ["syslog", "/var/log/nginx/access.log", "worker.log", "cron.log", "kern.log", "api.log"],
    "dir": ["~/Desktop/screenshots", "/var/www/html", "./src/components", "~/Library/Application Support",
            "/Volumes/Backup", "node_modules", "~/.config", "./dist", "/private/tmp", "~/Downloads/old"],
    "dir2": ["/Volumes/USB", "~/Archive/2024", "../backup", "~/Dropbox", "/srv/data"],
    "proc": ["mysqld", "httpd", "Finder", "Dock", "WindowServer", "coreaudiod", "mdworker", "kernel_task", "node"],
    "host": ["db.internal", "192.168.0.100", "example.net", "10.1.2.3", "router.local", "staging.example.com"],
    "domain": ["reddit.com", "netflix.com", "amazon.com", "mozilla.org", "python.org", "rust-lang.org"],
    "pkg": ["httpie", "watch", "zsh", "starship", "lazygit", "k9s", "helm", "yt-dlp", "pandoc", "sqlite", "mosh"],
    "pypkg": ["sqlalchemy", "pydantic", "boto3", "celery", "uvicorn", "jupyter", "tqdm", "openai", "scipy"],
    "npmpkg": ["vue", "svelte", "tailwindcss", "mongoose", "socket.io", "moment", "chalk", "commander"],
    "image": ["traefik", "jenkins/jenkins", "postgres:16", "rabbitmq:3", "elasticsearch:8.11.0", "nginx:alpine"],
    "container": ["frontend", "backend", "cache", "queue", "proxy", "jenkins", "es"],
    "video": ["screencast.mov", "vlog.mp4", "talk.mkv", "gopro.mp4", "meeting.webm", "trailer.mov"],
    "audio": ["interview.wav", "beat.mp3", "lecture.m4a", "sample.flac", "call.ogg"],
    "img": ["diagram.png", "cover.jpg", "icon.svg", "scan.png", "selfie.heic", "chart.webp"],
    "branch": ["feature/search", "fix/memory-leak", "release/2.0", "chore/deps", "feature/onboarding", "v2"],
    "user": ["jenkins", "git", "postgres", "dev", "nobody", "carol"],
    "port": ["8443", "9090", "1883", "5601", "11434", "6006"],
    "word": ["panic", "exception", "deadline", "secret", "segfault", "unauthorized", "retry"],
    "url": ["https://example.org/api/v1/items", "http://127.0.0.1:8000/docs", "https://httpbin.org/post",
            "https://example.com/data.csv", "https://api.example.com/health"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n-comp", dest="n_comp", type=int, default=12000)
    ap.add_argument("--per-fam", dest="per_fam", type=int, default=70)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/draft/prompts4.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for k, v in EXTRA.items():
        G2.S[k] = list(dict.fromkeys(G2.S[k] + v))
    G.S.update(G2.S)
    files = ["prompts.jsonl", "prompts2.jsonl", "prompts3.jsonl"]
    old = [json.loads(l) for f in files for l in open(os.path.join(ROOT, "scratch/draft", f))]
    bench = [l for l in open(os.path.join(ROOT, "bench/prompts.txt")).read().split("\n") if l]
    bf = G.BenchFilter(bench)
    held = G2.NearDup([r["prompt"] for r in old if r["split"] == "heldout"])
    seen = {" ".join(G.words(r["prompt"])) for r in old} | {" ".join(G.words(b)) for b in bench}
    train_groups = {r["group"] for r in old if r["split"] == "train"}
    cand = []
    # 1. family expansions (TRAIN families only) with the extended slot values
    for fid, temps in G.F:
        if f"fam:{fid}" in train_groups:
            for b in G.expand_family(rng, temps, a.per_fam):
                cand.append((f"fam:{fid}", G.phrase(rng, b) if rng.random() < 0.6 else b))
    for fid, temps in G2.F2:
        for b in G.expand_family(rng, temps, a.per_fam):
            cand.append((f"fam2:{fid}", G.phrase(rng, b) if rng.random() < 0.6 else b))
    # 2. new phrasing variants of TRAIN hand-written bases
    for tag, H in (("hand", HAND), ("hand2", HAND2)):
        for area, block in H.items():
            if area in ("long", "terse", "casual"):
                continue
            for i, line in enumerate(l.strip() for l in block.strip().split("\n")):
                g = f"{tag}:{area}:{i}"
                if line and (tag == "hand2" or g in train_groups):
                    for _ in range(2):
                        cand.append((g, G.phrase(rng, line)))
    # 3. compositional requests
    tries = 0
    ncomp = 0
    while ncomp < a.n_comp and tries < a.n_comp * 6:
        tries += 1
        di = rng.choices(range(len(G3.DOMAINS)), weights=G3.WEIGHTS)[0]
        acts, objs, mods, place = G3.DOMAINS[di]
        act, obj, mod = rng.choice(acts), rng.choice(objs), rng.choice(mods)
        if not G3.compatible(di, act, obj, mod):
            continue
        base = f"{act} {obj}"
        if place and rng.random() < 0.7:
            base += " " + rng.choice(G3.PLACES)
        base = G3.fill(rng, base + mod)
        cand.append((f"comp:{di}:{act}", G.phrase(rng, base) if rng.random() < 0.5 else base))
        ncomp += 1
    recs = []
    rej = {"dup": 0, "bench": 0, "held": 0}
    for g, p in cand:
        p = re.sub(r"\s+", " ", p).strip()
        k = " ".join(G.words(p))
        if not k or k in seen:
            rej["dup"] += 1
            continue
        if bf.reject(p) or G.excluded(p):
            rej["bench"] += 1
            continue
        if held.reject(p):
            rej["held"] += 1
            continue
        seen.add(k)
        recs.append({"prompt": p, "group": g, "split": "train"})
    rng.shuffle(recs)
    base_id = max(r["id"] for r in old) + 1
    for i, r in enumerate(recs):
        r["id"] = base_id + i
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} new train prompts from {len(cand)} candidates; rejected {rej}")


if __name__ == "__main__":
    main()
