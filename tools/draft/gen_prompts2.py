"""Training-data expansion: more requests, all assigned to TRAIN.

Sources: (1) HAND2 lines (+ phrasing variants), (2) new template families F2, (3) extra
expansions of the TRAIN template families of gen_prompts.py, (4) extra phrasing variants of
TRAIN hand-written bases. Every new prompt is rejected if it duplicates an existing prompt,
is a near-duplicate of a bench prompt, or is a near-duplicate of any HELD-OUT prompt (so the
held-out numbers stay a test on unseen request families).

usage: .venv/bin/python -I tools/draft/gen_prompts2.py [--per-fam 60] [--seed 2]
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_prompts as G  # noqa: E402
from prompts_hand import HAND  # noqa: E402
from prompts_hand2 import HAND2  # noqa: E402

ROOT = G.ROOT
S = dict(G.S)
S.update({
    "height": ["480", "600", "768", "1080", "200"],
    "a": ["12", "250", "1024", "3.5", "99", "7"],
    "b": ["4", "16", "60", "8", "3", "1000"],
    "rate": ["16000", "44100", "48000", "22050"],
    "crate": ["serde", "tokio", "clap", "reqwest", "anyhow", "rayon"],
    "cargocmd": ["build --release", "test", "clippy", "fmt", "doc --open", "update", "bench"],
    "gomod": ["github.com/gin-gonic/gin", "golang.org/x/sync", "github.com/spf13/cobra"],
    "gofile": ["main.go", "cmd/server/main.go", "tool.go"],
    "goos": ["linux", "windows", "arm64 macs", "linux arm64"],
    "gopkg": ["internal/...", "pkg/api", "cmd/..."],
    "prnum": ["42", "128", "7", "1999"],
    "defaults_key": ["com.apple.dock autohide", "com.apple.finder AppleShowAllFiles", "NSGlobalDomain KeyRepeat",
                     "com.apple.screencapture disable-shadow"],
    "bool": ["true", "false", "2", "0"],
    "defaults_domain": ["com.apple.dock", "com.apple.finder", "NSGlobalDomain", "com.apple.Safari"],
    "service": ["Wi-Fi", "Ethernet", "Thunderbolt Bridge", "USB 10/100/1000 LAN"],
    "dnsip": ["1.1.1.1", "8.8.8.8 8.8.4.4", "9.9.9.9", "208.67.222.222"],
    "plist": ["~/Library/LaunchAgents/com.me.backup.plist", "/Library/LaunchDaemons/com.example.helper.plist",
              "com.user.sync.plist"],
    "label": ["com.apple", "homebrew", "backup", "docker"],
    "disk": ["disk2", "disk4", "/dev/disk3", "disk5s1"],
    "port2": ["80", "5432", "3306", "8080", "6379"],
    "bits": ["2048", "4096", "3072"],
    "email": ["me@example.com", "alice@company.com", "dev@team.io"],
    "prefix": ["logs/2024", "images", "backups/daily", "tmp"],
    "instance": ["i-0abc123def456", "i-0123456789abcdef0", "i-09f8e7d6c5b4"],
    "fn": ["my-function", "image-resizer", "api-handler"],
    "chart": ["bitnami/redis", "ingress-nginx/ingress-nginx", "prometheus-community/prometheus", "grafana/grafana"],
    "release": ["my-redis", "ingress", "monitoring", "web"],
    "svc": ["nginx", "docker", "ssh", "postgresql", "redis-server", "cron"],
    "video2": ["outro.mp4", "part2.mp4", "b-roll.mov", "intro.mp4"],
    "img2": ["right.png", "photo2.jpg", "overlay.png", "second.jpg"],
    "px": ["5", "10", "20", "2"],
    "colnum": ["2", "3", "4", "5", "1"],
    "word2": ["warning", "done", "localhost:8080", "true", "v2"],
    "tz": ["Tokyo", "London", "New York", "UTC", "Berlin", "Sydney", "India"],
    "datestr": ["2024-01-01", "2023-12-25 10:30", "March 3 2022", "2025-06-30"],
    "keyfile": ["~/.ssh/id_ed25519.pub", "~/.ssh/id_rsa", "deploy_key.pub"],
    "session": ["work", "dev", "logs", "main", "build"],
    "pid": ["1234", "4321", "999", "55012", "301"],
    "nice": ["10", "19", "-5", "5"],
    "file2": ["notes2.txt", "data2.csv", "image.png", "script.sh", "README.md"],
    "archive2": ["bundle.zip", "files.zip", "backup.zip", "photos.zip"],
    "dockerfile": ["Dockerfile.dev", "docker/Dockerfile", "Dockerfile.prod"],
    "tagname": ["myapp:dev", "api:1.0", "web:latest", "worker:test"],
    "cpus": ["1", "2", "0.5", "4"],
    "script_name": ["build", "dev", "lint", "test:watch", "start:prod", "format"],
    "ver": ["1.2.3", "4.17.21", "2.0.0", "18.2.0", "0.9"],
    "module": ["http.server", "venv", "pip", "json.tool", "unittest", "pytest", "timeit"],
    "pyfile": ["app.py", "manage.py", "train.py", "cli.py", "scrape.py"],
    "voice": ["Samantha", "Alex", "Daniel", "Karen"],
    "afile": ["speech.aiff", "hello.m4a", "message.aiff"],
    "clock": ["7:00", "6:30am", "08:15", "midnight"],
    "plistfile": ["Info.plist", "com.apple.dock.plist", "settings.plist", "config.plist"],
    "key": ["CFBundleVersion", "CFBundleIdentifier", "LSMinimumSystemVersion"],
    "dmgfile": ["installer.dmg", "MyApp.dmg", "backup.dmg", "tools.dmg"],
    "csvfile": ["sales.csv", "users.csv", "data.csv", "export.csv", "metrics.csv"],
    "field": ["name", "id", "email", "status", "price", "version"],
    "code": ["404", "500", "403", "502"],
    "ip": ["10.0.0.7", "192.168.0.12", "172.16.4.2", "8.8.4.4"],
})

F2 = [
    ("magick_resize", ["resize {img} to {width}x{height} with imagemagick", "make a {width}px thumbnail of {img} with magick", "convert {img} to {ifmt} with imagemagick"]),
    ("pdf_pages", ["extract pages {count} to {n} from {pdf}", "rotate {pdf} by 90 degrees", "remove the password from {pdf}", "convert {pdf} to text"]),
    ("sox", ["trim {audio} to {seconds} seconds with sox", "convert {audio} to mono with sox", "change the sample rate of {audio} to {rate}", "show the duration of {audio}"]),
    ("cargo", ["add the {crate} crate to my rust project", "run cargo {cargocmd}", "create a new rust library called {folder}"]),
    ("golang", ["go get {gomod}", "build {gofile} for {goos}", "run the go tests in ./{gopkg}", "run {gofile}"]),
    ("gh", ["create a github issue titled '{msg}'", "list pull requests by {user} with gh", "check out pull request #{prnum} with gh", "merge pull request {prnum} with squash using gh"]),
    ("defaults", ["set {defaults_key} to {bool} with defaults write", "read the {defaults_domain} defaults", "reset the {defaults_domain} preferences"]),
    ("networksetup", ["set the dns servers for {service} to {dnsip}", "turn the {service} service off", "show info about the {service} network service"]),
    ("launchctl", ["load the launch agent {plist}", "unload {plist}", "list launchd jobs matching {label}"]),
    ("diskutil", ["list the partitions of {disk}", "unmount {disk}", "erase {disk} as apfs named {volname}"]),
    ("ssh_tunnel", ["forward local port {port} to {host}:{port2} over ssh", "open a socks proxy on port {port} through {host}", "expose local port {port} on {host} with a reverse tunnel"]),
    ("openssl", ["show the certificate of {domain}", "check when the certificate for {domain} expires", "generate a {bits}-bit rsa key", "create a csr for {domain}"]),
    ("gpg", ["encrypt {file} for {email} with gpg", "decrypt {file}.gpg", "sign {file} with my gpg key", "verify the signature {file}.sig"]),
    ("aws2", ["list objects in s3://{bucket}/{prefix}", "delete s3://{bucket}/{prefix} recursively", "stop the ec2 instance {instance}", "tail the cloudwatch logs of /aws/lambda/{fn}"]),
    ("kubectl2", ["get the yaml of deployment {dep}", "port forward pod {pod} port {port}", "show the events for pod {pod}", "set the image of deployment {dep} to {image}"]),
    ("helm", ["install the {chart} helm chart as {release}", "uninstall the helm release {release}", "upgrade {release} with values.yaml"]),
    ("systemctl", ["restart the {svc} service", "show the status of {svc}", "enable {svc} at boot", "show the logs of {svc} since an hour ago"]),
    ("apt", ["install {pkg} on ubuntu", "remove {pkg} with apt", "show the version of {pkg} available in apt"]),
    ("ffmpeg2", ["add {audio} as the soundtrack of {video}", "overlay {img} on {video}", "concatenate {video} and {video2}", "extract every {n}th frame of {video}"]),
    ("imagemagick2", ["add a {px} pixel border to {img}", "make the white background of {img} transparent", "combine {img} and {img2} side by side"]),
    ("text_awk", ["print the {k} field of each line of {textfile} using awk", "sum column {colnum} of {textfile} with awk", "print lines of {textfile} where column {colnum} is greater than {n}"]),
    ("sed2", ["delete line {n} from {textfile}", "insert '{word}' at the start of every line of {textfile}", "replace the first occurrence of {word} with {word2} in {textfile}"]),
    ("find2", ["find .{ext} files in {dir} modified after {file}", "find directories named {folder} in {dir}", "find files in {dir} owned by {user}", "find files in {dir} with permission {mode}"]),
    ("xargs", ["delete the files listed in {textfile}", "run {script} on every .{ext} file in {dir} in parallel", "copy the files listed in {textfile} to {dir2}"]),
    ("datecalc", ["show the date {n} days ago", "show the current time in {tz}", "convert {datestr} to a unix timestamp"]),
    ("math", ["calculate {a} times {b} in the terminal", "what is {a} divided by {b} with decimals", "convert {a} to hexadecimal"]),
    ("ssh_key", ["generate an ssh key with the comment {email}", "copy my ssh key to {user}@{host}", "show the fingerprint of {keyfile}"]),
    ("tmux", ["create a tmux session called {session}", "kill the tmux session {session}", "attach to tmux session {session}"]),
    ("vim", ["open {file} in vim at line {n}", "open {file} read only in vim", "edit {file} with nano"]),
    ("ps2", ["show the cpu usage of pid {pid}", "kill process {pid}", "show the open files of pid {pid}", "renice process {pid} to {nice}"]),
    ("curl3", ["download {url} and save it as {file}", "send a DELETE request to {url}", "show the timing breakdown for {url}", "download {url} through a proxy"]),
    ("zip2", ["zip {file} and {file2} into {archive2}", "unzip {archive2} to {dir2}", "add {file} to {archive2}"]),
    ("brew2", ["show the dependencies of {pkg}", "pin {pkg} in homebrew", "show why {pkg} is installed"]),
    ("docker2", ["build a docker image from {dockerfile} tagged {tagname}", "run {image} with {dir} mounted at /data", "show the environment variables of the {container} container", "limit the {container} container to {cpus} cpus"]),
    ("git3", ["show the diff between {branch} and {branch2}", "show commits that touched {file} in the last {days} days", "restore {file} as it was {n} commits ago", "stash only {file}"]),
    ("npm3", ["run the {script_name} script with npm", "run the {script_name} script with yarn", "add {npmpkg}@{ver} as a dependency"]),
    ("py3", ["run the python {module} module", "pip install {pypkg}=={ver}", "run {pyfile} with the --verbose flag"]),
    ("mac_open", ["open {file} with {app}", "open {dir} in Finder", "reveal {file} in Finder"]),
    ("say2", ["speak '{phrase}' with the voice {voice}", "save '{phrase}' spoken aloud to {afile}"]),
    ("caffeinate", ["keep the mac awake for {n} minutes", "prevent sleep until {script} finishes"]),
    ("pmset", ["make the display sleep after {n} minutes", "schedule the mac to wake at {clock}", "show the current battery level"]),
    ("plist", ["convert {plistfile} to xml", "print the value of {key} in {plistfile}", "lint {plistfile}"]),
    ("hdiutil", ["create a dmg from the folder {folder}", "mount {dmgfile}", "convert {dmgfile} to a read-write image"]),
    ("csv2", ["count the rows in {csvfile}", "show the first {n} rows of {csvfile} as a table", "convert {csvfile} to json", "get the unique values of column {colnum} in {csvfile}"]),
    ("json2", ["get the {field} field from every object in {jsonfile}", "filter {jsonfile} for items where {field} is {val}", "count the items in {jsonfile}"]),
    ("log2", ["count the {code} errors in {logfile}", "show the requests from {ip} in {logfile}", "show the most requested urls in {logfile}"]),
    ("perm2", ["give the group write permission on {dir}", "remove execute permission from {file}", "make {dir} readable by everyone"]),
]


class NearDup:
    """Fast near-duplicate check against a reference set (inverted index on content words)."""

    def __init__(self, refs):
        self.refs = refs
        self.rw = [G.words(r) for r in refs]
        self.rc = [set(w) - G.BENCH_STOP for w in self.rw]
        self.inv = defaultdict(set)
        for i, c in enumerate(self.rc):
            for w in c:
                self.inv[w].add(i)

    def reject(self, p):
        w = G.words(p)
        c = set(w) - G.BENCH_STOP
        cnt = defaultdict(int)
        for x in c:
            for i in self.inv.get(x, ()):
                cnt[i] += 1
        for i, k in cnt.items():
            j = k / max(1, len(c | self.rc[i]))
            if j >= 0.5:
                return True
            if k >= 2 and difflib.SequenceMatcher(None, w, self.rw[i]).ratio() >= 0.6:
                return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fam", type=int, default=60)
    ap.add_argument("--old-fam-extra", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/draft/prompts2.jsonl"))
    a = ap.parse_args()
    G.S.update(S)  # expand_family reads G.S
    rng = random.Random(a.seed)
    old = [json.loads(l) for l in open(os.path.join(ROOT, "scratch/draft/prompts.jsonl"))]
    bench = [l for l in open(os.path.join(ROOT, "bench/prompts.txt")).read().split("\n") if l]
    bf = G.BenchFilter(bench)
    held = NearDup([r["prompt"] for r in old if r["split"] == "heldout"])
    seen = {" ".join(G.words(r["prompt"])) for r in old} | {" ".join(G.words(b)) for b in bench}
    train_groups = {r["group"] for r in old if r["split"] == "train"}

    cand = []  # (group, prompt)
    for area, block in HAND2.items():
        for i, line in enumerate(l.strip() for l in block.strip().split("\n")):
            if not line:
                continue
            g = f"hand2:{area}:{i}"
            cand.append((g, line))
            if area != "casual":
                for _ in range(rng.choice([1, 2, 2, 3])):
                    cand.append((g, G.phrase(rng, line)))
    for fid, temps in F2:
        for b in G.expand_family(rng, temps, a.per_fam):
            cand.append((f"fam2:{fid}", G.phrase(rng, b) if rng.random() < 0.55 else b))
    for fid, temps in G.F:
        g = f"fam:{fid}"
        if g not in train_groups:
            continue
        for b in G.expand_family(rng, temps, a.old_fam_extra + 40):
            cand.append((g, G.phrase(rng, b) if rng.random() < 0.6 else b))
    for area, block in HAND.items():
        if area in ("long", "terse"):
            continue
        for i, line in enumerate(l.strip() for l in block.strip().split("\n")):
            g = f"hand:{area}:{i}"
            if line and g in train_groups:
                for _ in range(2):
                    cand.append((g, G.phrase(rng, line)))

    recs = []
    nrej = {"dup": 0, "bench": 0, "held": 0}
    for g, p in cand:
        p = re.sub(r"\s+", " ", p).strip()
        k = " ".join(G.words(p))
        if not k or k in seen:
            nrej["dup"] += 1
            continue
        if bf.reject(p) or G.excluded(p):
            nrej["bench"] += 1
            continue
        if held.reject(p):
            nrej["held"] += 1
            continue
        seen.add(k)
        recs.append({"prompt": p, "group": g, "split": "train"})
    rng.shuffle(recs)
    base = max(r["id"] for r in old) + 1
    for i, r in enumerate(recs):
        r["id"] = base + i
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} new train prompts from {len(cand)} candidates; rejected {nrej}")


if __name__ == "__main__":
    main()
