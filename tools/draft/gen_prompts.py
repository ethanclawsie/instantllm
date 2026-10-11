"""Generate a diverse set of shell-command requests for drafter training/evaluation.

Writes scratch/draft/prompts.jsonl: {"id", "prompt", "group", "split"} where group is the
base request (hand-written line or template family) and split is "train" or "heldout".
The split is by group, so held-out requests come from families and base requests never
seen in training. Bench prompts and near-duplicates of them are rejected.

usage: .venv/bin/python -I tools/draft/gen_prompts.py [--target 6000] [--seed 1]
"""
import argparse
import difflib
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_hand import HAND  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ----------------------------------------------------------------------------------------
# slot values
S = {
    "ext": ["py", "js", "ts", "swift", "go", "rs", "java", "c", "cpp", "h", "rb", "php", "html",
            "css", "json", "yaml", "yml", "md", "txt", "csv", "log", "pdf", "jpg", "png", "gif",
            "mp4", "mov", "mp3", "wav", "zip", "sh", "sql", "xml", "toml", "conf", "ipynb", "kt",
            "lua", "vue", "tsx", "jsx", "svg", "webp", "heic", "dmg", "bak", "tmp", "o", "class"],
    "lang": ["python", "javascript", "typescript", "swift", "go", "rust", "java", "c", "c++",
             "ruby", "php", "kotlin", "shell", "markdown", "yaml", "json", "html", "css", "lua"],
    "dir": ["the current directory", "my home directory", "Downloads", "~/Downloads", "Desktop",
            "~/Documents", "/tmp", "/var/log", "~/Projects", "src", "the project folder",
            "/usr/local", "/etc", "~/Library/Caches", "~/Pictures", "~/Movies", "./build",
            "this folder", "the repo", "/opt/homebrew", "~/code", "~/Music", "assets", "docs"],
    "dir2": ["~/Backup", "/Volumes/External", "archive", "old", "~/Documents/sorted", "/tmp/out",
             "backup", "~/Desktop/stuff", "dist"],
    "size": ["1MB", "10MB", "50MB", "500MB", "1GB", "2GB", "5 gigabytes", "200k", "20 megabytes"],
    "n": ["3", "5", "10", "15", "20", "25", "30", "100"],
    "days": ["1", "2", "3", "14", "30", "60", "90", "365"],
    "proc": ["python", "chrome", "Safari", "Slack", "docker", "java", "ruby", "nginx", "postgres",
             "redis-server", "Xcode", "Spotify", "zoom.us", "firefox", "Mail", "code", "ollama"],
    "app": ["Safari", "Slack", "Spotify", "Xcode", "Mail", "Calendar", "Notes", "Terminal",
            "Google Chrome", "Firefox", "Music", "Photos", "Visual Studio Code", "Zoom"],
    "port": ["3000", "5432", "8000", "8888", "443", "80", "22", "6379", "5000", "9000", "27017",
             "4200", "8081", "5173", "3306"],
    "host": ["google.com", "github.com", "8.8.8.8", "192.168.1.1", "example.org", "api.example.com",
             "localhost", "10.0.0.5", "1.1.1.1", "apple.com", "my-server.local", "cloudflare.com"],
    "domain": ["example.com", "github.com", "google.com", "openai.com", "wikipedia.org",
               "mysite.dev", "apple.com", "stackoverflow.com"],
    "url": ["https://example.org/data.json", "https://api.github.com/users/octocat",
            "http://localhost:3000/health", "https://httpbin.org/get", "https://example.net/archive.tar.gz",
            "https://raw.githubusercontent.com/user/repo/main/install.sh", "https://example.com/report.pdf"],
    "branch": ["main", "develop", "release/1.2", "hotfix/crash", "bugfix/typo", "feature/payments",
               "dev", "staging", "feature/dark-mode", "experiment"],
    "branch2": ["main", "develop", "master", "staging", "release"],
    "file": ["config.yaml", "notes.txt", "data.csv", "report.pdf", "index.html", "main.py",
             "app.js", "Dockerfile", "README.md", "package.json", ".env", "server.log",
             "backup.sql", "settings.json", "todo.txt", "Makefile", "styles.css", "main.go"],
    "textfile": ["notes.txt", "data.csv", "server.log", "access.log", "words.txt", "input.txt",
                 "list.txt", "results.tsv", "error.log", "names.txt", "output.log"],
    "logfile": ["server.log", "access.log", "error.log", "app.log", "/var/log/install.log",
                "debug.log", "nginx/error.log", "build.log"],
    "word": ["error", "TODO", "FIXME", "password", "deprecated", "timeout", "localhost", "import",
             "warning", "null", "api_key", "failed"],
    "user": ["alice", "bob", "admin", "ubuntu", "deploy", "root", "pi", "ec2-user"],
    "author": ["alice", "bob", "Jane Doe", "me", "john@example.com"],
    "pkg": ["wget", "htop", "jq", "tree", "ripgrep", "git", "ffmpeg", "imagemagick", "postgresql",
            "redis", "neovim", "tmux", "fzf", "gh", "kubectl", "terraform", "awscli", "coreutils",
            "nmap", "bat", "fd", "go", "rust", "python@3.12", "node"],
    "cask": ["firefox", "google-chrome", "visual-studio-code", "iterm2", "docker", "slack",
             "spotify", "vlc", "rectangle", "obsidian"],
    "npmpkg": ["express", "react", "lodash", "typescript", "eslint", "prettier", "axios", "jest",
               "nodemon", "webpack", "vite", "dotenv", "zod", "next"],
    "pypkg": ["requests", "numpy", "pandas", "flask", "django", "pytest", "black", "matplotlib",
              "scikit-learn", "torch", "fastapi", "httpx", "rich", "pyyaml"],
    "image": ["nginx", "postgres:15", "redis", "ubuntu:22.04", "python:3.11-slim", "node:20",
              "alpine", "mysql:8", "mongo", "grafana/grafana"],
    "container": ["web", "db", "api", "my-app", "redis-cache", "worker", "nginx", "postgres"],
    "video": ["video.mp4", "clip.mov", "movie.mkv", "lecture.mp4", "input.avi", "trip.mov",
              "recording.webm", "demo.mp4", "interview.mkv"],
    "vfmt": ["mp4", "webm", "mkv", "avi", "gif", "mov"],
    "afmt": ["mp3", "wav", "aac", "m4a", "flac", "ogg"],
    "audio": ["song.wav", "podcast.mp3", "voice.m4a", "track.flac", "memo.aac", "music.ogg"],
    "img": ["photo.jpg", "image.png", "screenshot.png", "logo.svg", "picture.heic", "banner.webp",
            "avatar.jpeg", "scan.tiff"],
    "ifmt": ["png", "jpg", "webp", "gif", "tiff", "bmp", "heic", "pdf"],
    "width": ["640", "800", "1024", "1280", "1920", "300"],
    "seconds": ["5", "10", "30", "60", "90", "120"],
    "archive": ["archive.zip", "backup.tar.gz", "source.tar.xz", "data.tgz", "files.tar.bz2",
                "project.7z", "release.tar"],
    "mode": ["644", "755", "600", "700", "640", "750"],
    "script": ["deploy.sh", "build.sh", "run.py", "install.sh", "backup.sh", "start.sh", "cleanup.sh"],
    "var": ["API_KEY", "NODE_ENV", "DEBUG", "PORT", "JAVA_HOME", "EDITOR", "GOPATH", "AWS_PROFILE"],
    "val": ["production", "1", "8080", "vim", "secret123", "/usr/local/go", "dev"],
    "pathdir": ["~/bin", "/opt/homebrew/bin", "~/.local/bin", "/usr/local/go/bin", "~/.cargo/bin"],
    "time": ["2am", "midnight", "6:30", "noon", "9pm", "every hour"],
    "msg": ["initial commit", "fix login bug", "update docs", "wip", "bump version", "refactor api"],
    "tag": ["v1.0.0", "v2.3.1", "release-2024", "v0.1", "beta"],
    "ns": ["default", "kube-system", "staging", "prod", "monitoring"],
    "k8s": ["pods", "services", "deployments", "configmaps", "secrets", "ingresses", "nodes"],
    "pod": ["web-5f6d", "api-0", "worker-abc12", "redis-master-0", "nginx-7c9"],
    "dep": ["web", "api", "worker", "frontend", "backend"],
    "bucket": ["my-bucket", "backups", "assets-prod", "logs-archive", "data-lake"],
    "db": ["shop", "app", "analytics", "users", "blog", "inventory"],
    "table": ["users", "orders", "products", "events", "sessions"],
    "res": ["memory", "cpu", "disk", "swap", "network"],
    "phrase": ["hello", "build complete", "time for a break", "tests passed", "wake up"],
    "browser": ["Safari", "Chrome", "Firefox", "the default browser"],
    "pct": ["10", "25", "50", "75", "100"],
    "iface": ["en0", "en1", "wi-fi", "the ethernet adapter"],
    "doc": ["report.docx", "slides.pptx", "notes.md", "page.html", "letter.odt"],
    "pdf": ["a.pdf", "chapter1.pdf", "invoice.pdf", "scan.pdf"],
    "pdf2": ["b.pdf", "chapter2.pdf", "receipt.pdf", "scan2.pdf"],
    "rec": ["A", "AAAA", "MX", "TXT", "NS", "CNAME", "SOA"],
    "fname": ["config.json", "Makefile", ".env", "id_rsa", "package.json", "docker-compose.yml",
              "settings.py", "Info.plist", ".gitignore", "README.md"],
    "folder": ["images", "backup", "old_files", "sorted", "pdfs", "archive"],
    "k": ["first", "second", "third", "fourth"],
    "ts": ["1700000000", "1600000000", "1234567890", "1714000000"],
    "alias": ["gs", "ll", "dc", "k", "py", "gco"],
    "cmd": ["git status", "ls -la", "docker compose", "kubectl", "python3", "git checkout"],
    "rdir": ["/home/ubuntu", "/var/www", "~/uploads", "/tmp", "/srv/app"],
    "sqlite": ["app.db", "data.sqlite", "test.db", "local.db"],
    "volname": ["DATA", "BACKUP", "USB", "MEDIA"],
    "lines": ["10", "20", "100", "200", "1000"],
    "count": ["2", "3", "4", "5", "10"],
    "fps": ["24", "25", "30", "60", "15"],
    "res2": ["480p", "720p", "1080p", "640x360", "1280x720"],
    "start": ["00:00:10", "00:01:00", "00:05:30", "0:30", "00:12:00"],
    "dur": ["10", "30", "45", "60", "15"],
    "pyver": ["3.10", "3.11", "3.12", "3.9"],
    "nodever": ["16", "18", "20", "22", "lts"],
    "env": ["venv", ".venv", "env", "myenv"],
    "cnt": ["2", "3", "4", "5", "8"],
    "jsonfile": ["package.json", "data.json", "config.json", "response.json", "tsconfig.json", "users.json"],
    "repo": ["https://github.com/user/project.git", "git@github.com:acme/api.git", "https://gitlab.com/team/app.git",
             "https://github.com/torvalds/linux", "git@bitbucket.org:me/tools.git"],
}

# ----------------------------------------------------------------------------------------
# template families: (family_id, [templates]) -- each template uses {slot} placeholders
F = [
    ("audio_extract", ["extract the audio from {video} as {afmt}", "get just the sound from {video} and save it as {afmt}", "rip the audio track of {video} to {afmt}"]),
    ("video_convert", ["convert {video} to {vfmt}", "turn {video} into a {vfmt} file", "transcode {video} to {vfmt} using ffmpeg"]),
    ("audio_convert", ["convert {audio} to {afmt}", "change {audio} into {afmt} format"]),
    ("video_trim", ["trim {video} to the first {seconds} seconds", "cut the first {seconds} seconds off {video}", "keep only {dur} seconds of {video} starting at {start}"]),
    ("video_scale", ["resize {video} to {res2}", "downscale {video} to {res2}", "make {video} {res2}"]),
    ("video_fps", ["change the frame rate of {video} to {fps} fps", "convert {video} to {fps} frames per second"]),
    ("video_gif", ["make a gif from {video}", "convert the first {seconds} seconds of {video} to a gif"]),
    ("video_mute", ["remove the audio from {video}", "mute {video} and save as silent.mp4"]),
    ("video_info", ["show the resolution and duration of {video}", "print the codec of {video} with ffprobe"]),
    ("video_thumb", ["save a thumbnail of {video} at {start}", "extract the frame at {start} from {video} as a png"]),
    ("img_convert", ["convert {img} to {ifmt}", "change {img} to {ifmt} format", "save {img} as a {ifmt}"]),
    ("img_resize", ["resize {img} to {width} pixels wide", "scale {img} down to a width of {width}", "make {img} {width}px wide keeping the aspect ratio"]),
    ("img_batch", ["convert all {ifmt} images in {dir} to jpg", "resize every png in {dir} to {width} pixels wide"]),
    ("brew_install", ["install {pkg} with homebrew", "brew install {pkg}", "how do I install {pkg} on my mac", "get {pkg} via brew"]),
    ("brew_uninstall", ["uninstall {pkg} using brew", "remove the {pkg} formula", "get rid of {pkg} installed with homebrew"]),
    ("brew_info", ["show info about {pkg} in homebrew", "which version of {pkg} does brew have", "list the files brew installed for {pkg}"]),
    ("brew_cask", ["install {cask} with brew", "install the {cask} app using homebrew cask", "uninstall the cask {cask}"]),
    ("pip_install", ["install {pypkg} with pip", "pip install {pypkg} into the current venv", "add the python package {pypkg}", "upgrade {pypkg} to the latest version"]),
    ("pip_show", ["print the version of {pypkg} with pip show", "uninstall {pypkg} with pip", "show where {pypkg} is installed"]),
    ("npm_install", ["install {npmpkg} with npm", "add {npmpkg} as a dev dependency", "install {npmpkg} globally", "remove {npmpkg} from the project"]),
    ("npm_misc", ["show the installed version of {npmpkg}", "update {npmpkg} to the latest version with npm", "add {npmpkg} using yarn", "install {npmpkg} with pnpm"]),
    ("ssh", ["ssh into {host} as {user}", "connect to {host} with ssh as user {user}", "open an ssh session to {user}@{host} on port 2222"]),
    ("scp", ["copy {file} to {user}@{host}:{rdir}", "upload {file} to {rdir} on {host} with scp", "download {rdir}/{file} from {host}"]),
    ("rsync", ["sync {dir} to {dir2} with rsync", "rsync {dir} to {user}@{host}:{rdir}", "back up {dir} to {dir2} showing progress"]),
    ("ping", ["ping {host} {count} times", "check if {host} is reachable", "ping {host} every {count} seconds"]),
    ("dns", ["look up the {rec} records for {domain}", "query the {rec} record of {domain} with dig", "show the {rec} records of {domain} using nslookup"]),
    ("curl_headers", ["show the http headers of {url}", "fetch {url} and print the status code", "download {url} with curl"]),
    ("curl_json", ["get {url} and pretty print the json", "POST json to {url}", "call {url} with an authorization header"]),
    ("port_kill", ["kill the process using port {port}", "free up port {port}", "stop whatever is bound to port {port}"]),
    ("port_check", ["check if port {port} is open on {host}", "test the connection to {host} on port {port}"]),
    ("docker_run", ["run {image} in docker on port {port}", "start a {image} container in the background", "run {image} interactively and remove it after exit"]),
    ("docker_exec", ["open a shell inside the {container} container", "exec into container {container}", "run ls inside the {container} container"]),
    ("docker_logs", ["show the logs of the {container} container", "tail the last {lines} log lines of {container}", "follow the logs of container {container}"]),
    ("docker_ctl", ["stop the {container} container", "restart the {container} container", "remove the {container} container", "show the ip of the {container} container"]),
    ("docker_pull", ["pull the {image} image", "remove the {image} image", "show the size of the {image} image"]),
    ("git_switch", ["switch to the {branch} branch", "check out {branch}", "go to branch {branch}"]),
    ("git_delete", ["delete the local branch {branch}", "delete the remote branch {branch}", "force delete branch {branch}"]),
    ("git_merge", ["merge {branch} into {branch2}", "rebase {branch} onto {branch2}", "show commits in {branch} that are not in {branch2}"]),
    ("git_author", ["show commits by {author} in the last {days} days", "list commits by {author}", "count the commits made by {author}"]),
    ("git_commit", ["commit everything with the message \"{msg}\"", "stage all files and commit with message '{msg}'", "amend the last commit with the message \"{msg}\""]),
    ("git_tag", ["create a tag {tag} and push it", "delete the tag {tag} locally and on the remote", "check out the tag {tag}"]),
    ("git_log_n", ["show the last {n} commits", "show the last {n} commits with their diffs", "list the last {n} commit messages only"]),
    ("git_file", ["show the git history of {file}", "show who changed {file} last", "restore {file} from the last commit", "unstage {file}"]),
    ("find_name", ["find all files named {fname} under {dir}", "locate every {fname} inside {dir}", "search {dir} for a file called {fname}"]),
    ("find_delete", ["delete all .{ext} files in {dir}", "remove every .{ext} file under {dir}", "clean up the .{ext} files in {dir}"]),
    ("find_move", ["move all .{ext} files from {dir} to {dir2}", "copy every .{ext} file in {dir} to {dir2}", "copy all .{ext} files into a folder called {folder}"]),
    ("find_count", ["count the .{ext} files in {dir}", "how many {lang} files are in {dir}", "total size of all .{ext} files in {dir}"]),
    ("find_size", ["delete files bigger than {size} in {dir}", "move files above {size} from {dir} to {dir2}"]),
    ("find_age", ["delete files older than {days} days in {dir}", "list files in {dir} not modified for {days} days"]),
    ("grep_dir", ["find which files under {dir} contain {word}", "grep {dir} recursively for {word} ignoring case", "list the files in {dir} mentioning {word}"]),
    ("grep_log", ["show lines containing {word} in {logfile}", "count occurrences of {word} in {logfile}", "show lines in {logfile} without {word}"]),
    ("head_file", ["show the first {lines} lines of {textfile}", "print lines {n} to {lines} of {textfile}", "print line {n} of {textfile}"]),
    ("sort_file", ["sort {textfile} by the {k} column", "sort {textfile} numerically in reverse", "remove duplicate lines from {textfile}", "count unique lines in {textfile}"]),
    ("archive_make", ["create a zip of {dir}", "back up {dir} into {archive}"]),
    ("archive_extract", ["extract {archive}", "unpack {archive} into {dir2}", "list the contents of {archive}"]),
    ("chmod", ["change the permissions of {file} to {mode}", "set {file} to mode {mode}", "make {script} executable", "make {script} executable only by me"]),
    ("du_dir", ["show the size of {dir}", "how much space does {dir} use", "show the {n} biggest folders in {dir}"]),
    ("venv", ["create a python virtual environment named {env}", "create a venv called {env} with python {pyver}", "activate the {env} virtual environment"]),
    ("py_run", ["run {script} and save the output to output.log", "run {script} in the background", "run {script} every {n} minutes"]),
    ("env_var", ["set the {var} environment variable to {val}", "print the value of {var}", "unset {var}", "make {var}={val} permanent in zsh"]),
    ("path_add", ["add {pathdir} to my PATH", "prepend {pathdir} to PATH for this session", "check whether {pathdir} is in my PATH"]),
    ("alias", ["create an alias {alias} for '{cmd}'", "add a permanent alias {alias} that runs {cmd}", "remove the alias {alias}"]),
    ("password", ["generate a {n} character random password", "make a random password with {n} characters including symbols"]),
    ("timestamp", ["convert the unix timestamp {ts} to a date", "what date is epoch {ts}"]),
    ("proc_find", ["find the pid of {proc}", "show all {proc} processes", "show how much memory {proc} is using", "count running {proc} processes"]),
    ("proc_kill2", ["force kill {proc}", "restart {proc}", "kill the oldest {proc} process"]),
    ("app_ctl", ["quit {app}", "open {app}", "force quit {app}", "hide {app}"]),
    ("res_top", ["show which processes use the most {res}", "show {res} usage", "sort processes by {res}"]),
    ("open_url", ["open {url} in {browser}", "open {domain} in {browser}"]),
    ("volume", ["set the volume to {pct} percent", "set the volume to {pct}% from the terminal"]),
    ("cron", ["schedule {script} to run every day at {time} with cron", "add a cron job that runs {script} at {time}", "run {script} every {n} minutes using launchd"]),
    ("say", ["make the mac say '{phrase}'", "have the computer speak \"{phrase}\" out loud", "show a notification saying {phrase}"]),
    ("doc_convert", ["convert {doc} to pdf", "convert {doc} to markdown with pandoc", "count the words in {doc}"]),
    ("pdf_merge", ["merge {pdf} and {pdf2} into one pdf", "combine {pdf} and {pdf2} into combined.pdf", "split {pdf} into separate pages"]),
    ("screenshot_dir", ["change the screenshot save location to {dir2}", "save screenshots as {ifmt} files", "take a screenshot of a window and save it as {img}"]),
    ("k8s_list", ["list all {k8s} in the {ns} namespace", "get {k8s} in namespace {ns} as yaml", "watch the {k8s} in {ns}"]),
    ("k8s_pod", ["show logs for pod {pod}", "delete pod {pod}", "describe pod {pod} in {ns}", "exec into pod {pod}"]),
    ("k8s_scale", ["scale deployment {dep} to {count} replicas", "restart the {dep} deployment", "show the rollout status of {dep}"]),
    ("s3", ["upload {file} to s3 bucket {bucket}", "list the files in the s3 bucket {bucket}", "download everything from s3://{bucket} to {dir2}"]),
    ("pg", ["connect to postgres database {db} as {user}", "dump the {db} database to {db}.sql", "list the tables in the {db} postgres database", "drop the {db} database"]),
    ("sqlite", ["run the sql query 'select count(*) from {table}' on {sqlite}", "show the schema of {table} in {sqlite}", "export the {table} table from {sqlite} to csv"]),
    ("mysql", ["back up the mysql database {db}", "import {db}.sql into mysql", "show the tables of mysql database {db}"]),
    ("redis", ["get the key {table}:1 from redis", "list all redis keys matching {table}:*", "delete the redis key {table}:42"]),
    ("disk_fmt", ["format the usb drive as exfat named {volname}", "eject the volume {volname}", "show info about the {volname} volume"]),
    ("hash", ["compute the sha256 of {file}", "show the md5 of {archive}", "compare the checksums of {file} and {file}.bak"]),
    ("wc", ["count the lines in {textfile}", "count the words in {textfile}", "count the characters in {textfile}"]),
    ("lang_files", ["list all {lang} files in {dir}", "find the largest {lang} file in the repo"]),
    ("node_ver", ["switch to node {nodever} with nvm", "install node {nodever} using nvm", "make node {nodever} the default with nvm"]),
    ("py_ver", ["install python {pyver} with pyenv", "make a venv using python {pyver}", "run a script with python {pyver}"]),
    ("diff_files", ["diff {file} and {file}.orig", "show the differences between {textfile} and old_{textfile}", "compare {dir} and {dir2}"]),
    ("watch_cmd", ["run 'ls -la {dir}' every {count} seconds", "watch the size of {dir} change", "monitor {logfile} for new lines containing {word}"]),
    ("iface", ["show the ip address of {iface}", "show the mac address of {iface}", "turn {iface} off and on again"]),
    ("hosts_file", ["block {domain} using the hosts file", "map {domain} to 127.0.0.1", "remove {domain} from /etc/hosts"]),
    ("tar_exclude", ["create {archive} from {dir} excluding node_modules", "archive {dir} without the .git folder"]),
    ("jq_file", ["print the keys of {jsonfile} with jq", "pretty print {jsonfile}", "extract the .version field from {jsonfile}"]),
    ("csv_col", ["print the {k} column of {textfile}", "sum the {k} column of {textfile}", "show the {k} column of {textfile} sorted and unique"]),
    ("perm_owner", ["change the owner of {file} to {user}", "give {user} read access to {file}", "show the owner of {file}"]),
    ("ln", ["create a symlink to {file} in {dir2}", "link {pathdir} into /usr/local/bin"]),
    ("touch_mk", ["create a folder called {folder} in {dir}", "create {count} empty files named file1.txt to file{count}.txt", "make the directories {folder}/a/b/c"]),
    ("yt", ["download the youtube video https://youtu.be/dQw4w9WgXcQ as {vfmt}", "download https://www.youtube.com/watch?v=abc123 audio only as {afmt}"]),
    ("speed", ["make {video} play at 2x speed", "slow {video} down to half speed", "reverse {video}"]),
    ("crop", ["crop {img} to a 500x500 square", "rotate {img} by 90 degrees", "flip {img} horizontally", "convert {img} to grayscale"]),
    ("compose", ["docker compose up the {container} service", "rebuild the {container} service with compose", "show logs of the {container} compose service"]),
    ("git_remote", ["add {repo} as a remote called upstream", "change origin to {repo}", "clone {repo} into {folder}"]),
    ("tail_follow", ["watch {logfile} for errors", "count lines in {logfile} per hour"]),
]

# phrasing wrappers applied to a base imperative request
PREFIXES = [("", 50), ("how do I ", 8), ("how to ", 6), ("command to ", 6), ("what's the command to ", 4),
            ("show me how to ", 3), ("I need to ", 4), ("I want to ", 4), ("can you ", 3), ("please ", 3),
            ("help me ", 2), ("zsh command to ", 2), ("terminal command to ", 2), ("quick way to ", 2),
            ("one-liner to ", 2), ("how can I ", 4), ("give me a command to ", 2), ("mac: ", 1)]
SUFFIXES = [("", 70), ("?", 6), (" on my mac", 4), (" in the terminal", 4), (" from the command line", 3),
            (" using zsh", 2), (" please", 2), (" quickly", 1), (" on macos", 2), (".", 4)]

BENCH_STOP = set("the a an all in of to and my it for with on is this by as".split())


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def wchoice(rng, items):
    tot = sum(w for _, w in items)
    r = rng.uniform(0, tot)
    for v, w in items:
        r -= w
        if r <= 0:
            return v
    return items[-1][0]


QWORDS = ("how ", "what", "which", "where", "who", "is ", "are ", "why", "when", "does ", "can ")


def phrase(rng, base):
    if base.lower().startswith(QWORDS):
        return base + ("?" if rng.random() < 0.5 else "")
    p = wchoice(rng, PREFIXES)
    s = wchoice(rng, SUFFIXES)
    if p.startswith(("how", "what", "can", "is")) and s == "":
        s = "?" if rng.random() < 0.5 else ""
    if s == "?" and not p.startswith(("how", "what", "can")):
        s = ""
    b = base
    if p and b[:1].isupper() and not b[:2].isupper():
        b = b[0].lower() + b[1:]
    out = p + b + s
    r = rng.random()
    if r < 0.12:
        out = out[0].upper() + out[1:]
    return out


def expand_family(rng, templates, cap):
    outs = set()
    tries = 0
    while len(outs) < cap and tries < cap * 20:
        tries += 1
        t = rng.choice(templates)
        slots = re.findall(r"\{(\w+)\}", t)
        vals = {}
        for sl in slots:
            if sl not in vals:
                vals[sl] = rng.choice(S[sl])
        o = t.format(**vals)
        if not excluded(o):
            outs.add(o)
    return sorted(outs)


EXCLUDE = [
    r"\bmov\b.*\bmp4\b",                      # bench: convert video.mov to mp4
    r"\b(\w+)\.(\w+) (to|into a|as a|into) \2\b",  # same-format conversions
    r"tar\.gz",                                    # bench: compress a folder into a tar.gz
    r"\bjpe?g\b.*\bjpg\b",                      # bench: rename .jpeg to .jpg
    r"listening on port",
]


def excluded(p):
    return any(re.search(x, p) for x in EXCLUDE)


class BenchFilter:
    def __init__(self, bench):
        self.bench = bench
        self.bw = [words(b) for b in bench]
        self.bc = [set(w) - BENCH_STOP for w in self.bw]

    def score(self, p):
        w = words(p)
        c = set(w) - BENCH_STOP
        best = (0.0, 0.0, -1)
        for i, (bw, bc) in enumerate(zip(self.bw, self.bc)):
            r = difflib.SequenceMatcher(None, w, bw).ratio()
            j = len(c & bc) / max(1, len(c | bc))
            if max(r, j) > max(best[0], best[1]):
                best = (r, j, i)
        return best

    def reject(self, p):
        r, j, _ = self.score(p)
        return r >= 0.6 or j >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heldout", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/draft/prompts.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench = [l for l in open(os.path.join(ROOT, "bench/prompts.txt")).read().split("\n") if l]
    bf = BenchFilter(bench)

    groups = []  # (group_id, [prompts])
    # hand-written: each line is a group; add 0-2 phrasing variants
    for area, block in HAND.items():
        for i, line in enumerate(l.strip() for l in block.strip().split("\n")):
            if not line:
                continue
            g = f"hand:{area}:{i}"
            ps = {line}
            if area not in ("long", "terse"):
                for _ in range(rng.choice([0, 1, 1, 2])):
                    ps.add(phrase(rng, line))
            groups.append((g, sorted(ps)))
    n_hand = sum(len(p) for _, p in groups)
    # families: fill the remaining budget evenly
    remaining = max(0, a.target - n_hand)
    per = max(8, remaining // len(F))
    for fid, temps in F:
        base = expand_family(rng, temps, per)
        ps = set()
        for b in base:
            ps.add(phrase(rng, b) if rng.random() < 0.55 else b)
        groups.append((f"fam:{fid}", sorted(ps)))

    # dedupe (normalized), bench filter
    seen = set()
    bench_norm = {" ".join(words(b)) for b in bench}
    rejected = []
    clean = []
    for g, ps in groups:
        keep = []
        for p in ps:
            p = re.sub(r"\s+", " ", p).strip()
            k = " ".join(words(p))
            if not k or k in seen or k in bench_norm:
                continue
            if bf.reject(p) or excluded(p):
                rejected.append(p)
                continue
            seen.add(k)
            keep.append(p)
        if keep:
            clean.append((g, keep))

    # group split
    rng2 = random.Random(a.seed + 7)
    order = list(range(len(clean)))
    rng2.shuffle(order)
    total = sum(len(ps) for _, ps in clean)
    held = set()
    nh = 0
    for gi in order:
        if nh >= a.heldout * total:
            break
        # keep families from dominating: allow a family only if it does not overshoot by much
        sz = len(clean[gi][1])
        if nh + sz > a.heldout * total * 1.05:
            continue
        held.add(gi)
        nh += sz
    recs = []
    for gi, (g, ps) in enumerate(clean):
        for p in ps:
            recs.append({"prompt": p, "group": g, "split": "heldout" if gi in held else "train"})
    rng2.shuffle(recs)
    for i, r in enumerate(recs):
        r["id"] = i
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ntr = sum(r["split"] == "train" for r in recs)
    print(f"{len(recs)} prompts ({ntr} train, {len(recs) - ntr} heldout), {len(clean)} groups, "
          f"{n_hand} hand-written incl. variants, {len(rejected)} rejected as bench near-duplicates")
    # closest survivors per bench prompt (for the report)
    best = {}
    for r in recs:
        rr, jj, bi = bf.score(r["prompt"])
        s = max(rr, jj)
        if bi >= 0 and s > best.get(bi, (0, ""))[0]:
            best[bi] = (s, r["prompt"])
    with open(os.path.join(os.path.dirname(a.out), "bench_nearest.txt"), "w") as f:
        for bi, b in enumerate(bench):
            s, p = best.get(bi, (0, ""))
            f.write(f"{s:.2f}\t{b}\t{p}\n")
    with open(os.path.join(os.path.dirname(a.out), "bench_rejected.txt"), "w") as f:
        f.write("\n".join(rejected) + "\n")


if __name__ == "__main__":
    main()
