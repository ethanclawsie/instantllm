"""Training-data expansion #2: compositional requests (action x object x place x modifier)
across many domains, all assigned to TRAIN, filtered like gen_prompts2 (no duplicates, no
bench near-duplicates, no near-duplicates of HELD-OUT prompts).

usage: .venv/bin/python -I tools/draft/gen_prompts3.py [--n 14000] [--seed 3]
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

ROOT = G.ROOT

PLACES = ["in this folder", "in the current directory", "under ~/code", "in ~/Downloads", "on my desktop",
          "in /tmp", "in the repo", "inside src", "under /var/log", "in ~/Documents/work", "in my home folder",
          "in the parent directory", "under ./build", "in /usr/local/bin", "in ~/Pictures/2023",
          "inside the project", "in all subfolders", "in ~/Library/Logs", "in /etc", "on the external drive"]
FILE_OBJS = ["all {ext} files", "every .{ext} file", "files ending in .{ext}", "hidden files", "empty folders",
             "empty files", "files changed today", "the newest file", "the oldest file", "duplicate photos",
             "symlinks", "log files older than a week", "files without an extension", "zero byte files",
             "files with spaces in the name", "executable files", "files owned by root", "temporary files",
             "backup files ending in ~", "files named {fname}", "directories called {folder}", "large videos",
             "screenshots", "pdfs", "images", "csv files", "json files", "dotfiles", "files over {size}",
             "files under {size}", "files last opened a year ago", "read-only files", "git repositories"]
FILE_ACTS = ["list", "find", "delete", "count", "show the size of", "copy {dir2} with", "move to {dir2}",
             "archive", "show", "print the paths of", "compress", "open", "show the permissions of",
             "show the modification dates of", "sort by size", "make a list of", "back up", "rename", "touch",
             "print the names of", "remove", "search for", "locate"]
FILE_MODS = ["", "", "", " recursively", " sorted by size", " and show their sizes", " excluding node_modules",
             " in human readable form", " without asking for confirmation", " and print the count", " one per line",
             " with full paths", " ignoring case", " newest first", " but skip .git", " and save the list to files.txt",
             " quietly", " using find", " with ls", " in a single command", " and show the total"]

TEXT_OBJS = ["{textfile}", "the output of ps aux", "{logfile}", "my zsh history", "{csvfile}", "the clipboard",
             "{jsonfile}", "every markdown file", "the git log", "/etc/hosts", "the output of the last command",
             "{file}"]
TEXT_ACTS = ["count the lines in", "show the unique lines of", "sort", "show the first {lines} lines of",
             "show the last {lines} lines of", "find lines containing {word} in", "remove blank lines from",
             "count the words in", "show the most common words in", "number the lines of", "reverse the lines of",
             "show duplicate lines in", "lowercase", "uppercase", "trim whitespace in", "pretty print",
             "count characters in", "show column {colnum} of", "replace commas with tabs in",
             "strip comments from", "extract emails from", "extract urls from", "extract ip addresses from",
             "show lines matching a regex in", "print every other line of", "search {word} in",
             "highlight {word} in", "count occurrences of {word} in", "print the longest line of"]
TEXT_MODS = ["", "", "", " ignoring case", " and save the result", " in place", " with line numbers",
             " and show counts", " numerically", " using awk", " using sed", " with grep", " in reverse order",
             " without duplicates", " and copy it to the clipboard"]

GIT = ["show", "list", "delete", "create", "undo", "compare", "squash", "rebase", "push", "pull", "stash",
       "revert", "tag", "cherry-pick", "reset", "blame", "count", "search", "clean", "archive", "rename"]
GIT_OBJS = ["the last {n} commits", "all local branches", "branch {branch}", "remote branches", "my staged changes",
            "untracked files", "the commit {sha}", "tags", "changes in {file}", "the stash", "commits by {author}",
            "merged branches", "the reflog", "commits since yesterday", "the current branch", "submodules",
            "conflicting files", "the remote origin", "file {file} history", "the diff with {branch2}",
            "commits that mention {word}", "large files in history", "the first commit"]
GIT_MODS = ["", "", " in git", " with git", " on the remote", " safely", " and push", " without losing work",
            " with a one line summary", " including dates", " interactively", " for this repo"]

DOCKER = ["list", "stop", "remove", "restart", "inspect", "show logs of", "prune", "build", "pull", "push",
          "exec into", "show stats for", "rename", "tag", "save", "export", "pause", "kill", "show ports of",
          "copy files out of", "show the size of", "update"]
DOCKER_OBJS = ["all containers", "the {container} container", "unused images", "dangling volumes", "the {image} image",
               "stopped containers", "every running container", "compose services", "the network {container}-net",
               "images older than a week", "containers named {container}", "the build cache", "all volumes"]
DOCKER_MODS = ["", "", " with docker", " using docker compose", " forcefully", " and follow output", " in the background",
               " without the cache", " as json", " with a filter", " on this machine"]

NET = ["ping", "trace", "check", "scan", "resolve", "download from", "curl", "show the latency to", "test the speed of",
       "show open connections to", "look up", "connect to", "show the certificate of", "monitor", "block", "unblock",
       "show the headers of", "fetch", "query", "measure", "time"]
NET_OBJS = ["{host}", "{domain}", "{url}", "port {port} on {host}", "the gateway", "the dns server", "my router",
            "the vpn", "localhost:{port}", "the api at {url}", "{ip}", "every host on my lan", "the wifi network"]
NET_MODS = ["", "", " from the terminal", " with curl", " using nc", " with dig", " once", " {count} times",
            " every {count} seconds", " verbosely", " over ipv6", " with a timeout of 5 seconds", " silently"]

PROC = ["show", "kill", "restart", "find", "count", "monitor", "pause", "resume", "renice", "trace", "list",
        "show the memory of", "show the cpu of", "show open files of", "show the threads of", "watch"]
PROC_OBJS = ["{proc}", "process {pid}", "all {proc} processes", "the process on port {port}", "zombie processes",
             "my background jobs", "the heaviest process", "processes owned by {user}", "everything named {proc}",
             "the oldest process", "python scripts", "node servers", "the finder", "the dock"]
PROC_MODS = ["", "", " gracefully", " forcefully", " in real time", " sorted by memory", " sorted by cpu",
             " every {count} seconds", " with full command lines", " as a tree", " on my mac"]

MEDIA = ["convert", "compress", "resize", "trim", "rotate", "crop", "split", "merge", "mute", "speed up",
         "slow down", "extract the audio of", "make a gif of", "add subtitles to", "watermark", "stabilize",
         "normalize the audio in", "remove metadata from", "get the duration of", "thumbnail", "reverse",
         "change the bitrate of", "loop", "fade in", "fade out", "grayscale", "flip", "concatenate"]
MEDIA_OBJS = ["{video}", "{audio}", "{img}", "all mp4 files", "every png", "the screen recording", "{pdf}",
              "all wav files", "the photos", "{video} and {video2}", "the clip", "every heic photo", "the podcast"]
MEDIA_MODS = ["", "", " with ffmpeg", " to {vfmt}", " to {afmt}", " to {ifmt}", " to {res2}", " at {fps} fps",
              " without re-encoding", " keeping quality", " in place", " using imagemagick", " with sips",
              " to half size", " losslessly", " for the web"]

PKG = ["install", "uninstall", "upgrade", "update", "pin", "list", "search for", "show info about", "reinstall",
       "show the version of", "show dependencies of", "clean up", "audit", "freeze", "downgrade"]
PKG_OBJS = ["{pkg}", "{cask}", "{npmpkg}", "{pypkg}", "{crate}", "all brew packages", "outdated npm packages",
            "pip packages", "global npm packages", "every cask", "python {pyver}", "node {nodever}", "rust",
            "the latest go", "yarn", "pnpm", "poetry", "pipx", "uv"]
PKG_MODS = ["", "", " with brew", " with npm", " with pip", " with pipx", " with cargo", " globally",
            " for this project", " as a dev dependency", " quietly", " without prompts", " from source"]

SYS = ["show", "check", "set", "change", "reset", "turn off", "turn on", "toggle", "increase", "decrease",
       "print", "monitor", "list", "enable", "disable"]
SYS_OBJS = ["the battery level", "the cpu temperature", "free memory", "disk space", "the uptime", "the hostname",
            "dark mode", "the volume", "the screen brightness", "bluetooth", "wifi", "the firewall",
            "the time zone", "the dns cache", "spotlight indexing", "time machine", "the dock size",
            "key repeat speed", "the login shell", "the swap usage", "gatekeeper", "the screensaver",
            "auto updates", "the sleep timer", "my path", "environment variables", "the macos version",
            "the serial number", "the ip address", "the mac address", "the kernel version", "the system log"]
SYS_MODS = ["", "", " on my mac", " from the terminal", " with defaults", " with pmset", " with networksetup",
            " with sysctl", " permanently", " for now", " for all users", " every minute"]

DOMAINS = [
    (FILE_ACTS, FILE_OBJS, FILE_MODS, True), (TEXT_ACTS, TEXT_OBJS, TEXT_MODS, False),
    (GIT, GIT_OBJS, GIT_MODS, False), (DOCKER, DOCKER_OBJS, DOCKER_MODS, False),
    (NET, NET_OBJS, NET_MODS, False), (PROC, PROC_OBJS, PROC_MODS, False),
    (MEDIA, MEDIA_OBJS, MEDIA_MODS, False), (PKG, PKG_OBJS, PKG_MODS, False), (SYS, SYS_OBJS, SYS_MODS, False),
]
WEIGHTS = [4, 2, 2, 1.2, 1.2, 1, 1.5, 1, 1.5]


VID = ("{video}", "all mp4 files", "the screen recording", "{video} and {video2}", "the clip")
AUD = ("{audio}", "all wav files", "the podcast")
IMG = ("{img}", "every png", "the photos", "every heic photo")
PKG_TOOL = {" with brew": ("{pkg}", "{cask}", "all brew packages", "every cask", "python {pyver}", "node {nodever}",
                           "rust", "the latest go", "yarn", "pnpm", "poetry", "pipx", "uv"),
            " with npm": ("{npmpkg}", "outdated npm packages", "global npm packages", "yarn", "pnpm"),
            " with pip": ("{pypkg}", "pip packages", "poetry", "uv"), " with pipx": ("{pypkg}", "poetry", "uv"),
            " with cargo": ("{crate}",), " as a dev dependency": ("{npmpkg}", "{pypkg}")}


def compatible(di, act, obj, mod):
    if di == 6:  # media
        if mod in (" to {vfmt}", " to {res2}", " at {fps} fps", " without re-encoding") and obj not in VID:
            return False
        if mod == " to {afmt}" and obj not in AUD + VID:
            return False
        if mod in (" to {ifmt}", " with sips", " using imagemagick") and obj not in IMG:
            return False
        if obj in IMG and act in ("trim", "mute", "speed up", "slow down", "extract the audio of", "add subtitles to",
                                  "stabilize", "normalize the audio in", "get the duration of", "loop", "fade in",
                                  "fade out", "concatenate", "reverse", "change the bitrate of"):
            return False
        if obj in AUD and act in ("resize", "crop", "rotate", "make a gif of", "watermark", "stabilize", "thumbnail",
                                  "grayscale", "flip", "add subtitles to"):
            return False
        if obj == "{pdf}" and act not in ("compress", "split", "merge", "rotate", "crop", "remove metadata from"):
            return False
    if di == 7 and mod in PKG_TOOL and obj not in PKG_TOOL[mod]:
        return False
    if di == 8 and mod.startswith(" with "):
        return False
    return True


def fill(rng, s):
    while True:
        m = re.search(r"\{(\w+)\}", s)
        if not m:
            return s
        key = m.group(1)
        if key == "sha":
            val = "".join(rng.choice("0123456789abcdef") for _ in range(7))
        else:
            val = rng.choice(G2.S[key])
        s = s[:m.start()] + val + s[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/draft/prompts3.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    old = [json.loads(l) for l in open(os.path.join(ROOT, "scratch/draft/prompts.jsonl"))]
    old2 = [json.loads(l) for l in open(os.path.join(ROOT, "scratch/draft/prompts2.jsonl"))]
    bench = [l for l in open(os.path.join(ROOT, "bench/prompts.txt")).read().split("\n") if l]
    bf = G.BenchFilter(bench)
    held = G2.NearDup([r["prompt"] for r in old if r["split"] == "heldout"])
    seen = {" ".join(G.words(r["prompt"])) for r in old + old2} | {" ".join(G.words(b)) for b in bench}
    recs = []
    tries = 0
    rej = {"dup": 0, "bench": 0, "held": 0}
    while len(recs) < a.n and tries < a.n * 6:
        tries += 1
        di = rng.choices(range(len(DOMAINS)), weights=WEIGHTS)[0]
        acts, objs, mods, place = DOMAINS[di]
        act, obj, mod = rng.choice(acts), rng.choice(objs), rng.choice(mods)
        if not compatible(di, act, obj, mod):
            continue
        base = f"{act} {obj}"
        if place and rng.random() < 0.7:
            base += " " + rng.choice(PLACES)
        base += mod
        base = fill(rng, base)
        p = G.phrase(rng, base) if rng.random() < 0.5 else base
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
        recs.append({"prompt": p, "group": f"comp:{di}:{act}", "split": "train"})
    base_id = max(r["id"] for r in old + old2) + 1
    for i, r in enumerate(recs):
        r["id"] = base_id + i
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} compositional train prompts ({tries} tries); rejected {rej}")


if __name__ == "__main__":
    main()
