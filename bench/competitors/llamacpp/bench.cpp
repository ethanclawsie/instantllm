// llama.cpp competitor harness for instantllm (Homebrew libllama 0.6.0, all layers on Metal).
//
// Build: bench/competitors/llamacpp/build.sh   Run from the repo root (default paths are relative).
//
// Modes (one TSV row per request on stdout, readable progress on stderr):
//   full    The KV cache is cleared for every request and the whole templated prompt is prefilled.
//   prefix  The prefix "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n" is decoded once
//           per context and kept. Each request first drops the KV cells after it
//           (llama_memory_seq_rm), then prefills only "{PROMPT}<|im_end|>\n<|im_start|>assistant\n".
//   cold    Spawns a fresh one-shot llama-completion process per request (greedy, -ngl 99,
//           non-conversation mode, prompt not echoed) and times spawn -> exit. The CLI always
//           appends " [end of text]\n" (on EOS) and "\n\n" to stdout and has no flag to drop them;
//           the harness checks for and strips exactly those, so the text column is the generated
//           text only. The CLI stops at llama.cpp's end-of-generation set, which contains both
//           protocol stop tokens.
//
// Metrics (full/prefix):
//   e2e_ms        prompt string -> final text: template + tokenize + KV reset + prefill + greedy
//                 decode + detokenize.
//   ttft_ms       prompt string -> first generated token id.
//   gen_tokens    sampled tokens including the stop token (151645/151643) when one is emitted, so
//                 at most n_predict (64); forward passes = 1 prefill + (gen_tokens - 1) decodes.
//   decode_tok_s  (gen_tokens - 1) / (t_last_token - t_first_token); nan when gen_tokens == 1.
// cold: e2e_ms = spawn -> exit, ttft_ms = spawn -> first stdout byte; prefill_tokens, gen_tokens and
//   decode_tok_s come from the CLI's own --perf counters (eval time excludes sampling).
//
// Settings: every tunable takes a comma-separated list; the harness loads the model once and runs
// the cartesian product, one fresh context per combination (variant column gets a /fa=..,nb=..
// suffix when there is more than one). Defaults = fastest settings measured on the base M5
// (q4_0 sweeps, 4 prompts x 3 reps per config, all four GGUF variants checked for FA/tensor API):
//   flash attention   on (auto resolves to on); off decodes ~5% slower on every variant
//                     (q4_0 255 vs 270 tok/s, q8_0 191 vs 201, fp16 117 vs 119).
//   n_batch/n_ubatch  512/512: no measurable effect while the prompt fits one ubatch (>= 64);
//                     n_ubatch 32 splits the prefill and adds ~10 ms TTFT.
//   threads           1: no effect in-process (1..8 identical, all layers on Metal); saves ~10 ms
//                     of threadpool start-up in cold mode.
//   sampler           CPU greedy chain; backend (on-GPU) greedy sampling is within noise.
//   Metal tensor API  (env GGML_METAL_TENSOR_DISABLE=1 turns it off; on by default for M5):
//                     prefill of 17..66 tokens costs a flat ~14 ms with it vs ~18 ms without
//                     (full prompts), but 17..22-token prefix-mode suffixes are ~1 ms faster
//                     without it; <= 16 tokens (~7 ms) equal. run_competitors.sh therefore
//                     disables it for prefix mode only.
//   cold CLI          -fit off saves ~180 ms (the fit step does a dry-run load), --no-warmup
//                     another ~15-60 ms; -c 512, -b/-ub 128..512, --backend-sampling: no effect.
//                     Greedy is "--samplers top_k --top-k 1": one candidate survives, so the
//                     chain's final dist sampler is deterministic argmax.

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <spawn.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char ** environ;

namespace {

constexpr llama_token kImEnd     = 151645; // <|im_end|>
constexpr llama_token kEndOfText = 151643; // <|endoftext|>
static bool g_ignore_eos = false;           // --ignore-eos: always generate n_predict tokens

const char * const kPrefixHead = "<|im_start|>system\n";
const char * const kPrefixTail = "<|im_end|>\n<|im_start|>user\n";
const char * const kSuffix     = "<|im_end|>\n<|im_start|>assistant\n";

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("bench: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read %s", path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> read_prompts(const std::string & path) {
    std::vector<std::string> out;
    std::stringstream ss(read_file(path));
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, sep)) {
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

std::vector<int> split_ints(const std::string & s) {
    std::vector<int> out;
    for (const auto & t : split(s, ',')) out.push_back(std::stoi(t));
    return out;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string tsv_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '\t': o += "\\t";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            default:   o += c;      break;
        }
    }
    return o;
}

std::string fmt(double v, int prec) {
    if (!std::isfinite(v)) return "nan";
    char b[64];
    snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}

std::string fmt_int(int v) { return v < 0 ? "nan" : std::to_string(v); }

double median(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double min_of(const std::vector<double> & v) {
    double m = NAN;
    for (double x : v) if (std::isfinite(x) && (!std::isfinite(m) || x < m)) m = x;
    return m;
}

struct Row {
    std::string engine = "llama.cpp", variant, mode;
    int prompt_idx = 0, rep = 0;
    double e2e = NAN, ttft = NAN, decode_tps = NAN;
    int n_prefill = -1, n_gen = -1;
    std::string text;
};

void print_header() {
    printf("engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext\n");
    fflush(stdout);
}

void print_row(const Row & r) {
    printf("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n", r.engine.c_str(), r.variant.c_str(), r.mode.c_str(),
           r.prompt_idx, r.rep, fmt(r.e2e, 3).c_str(), fmt(r.ttft, 3).c_str(), fmt_int(r.n_prefill).c_str(),
           fmt_int(r.n_gen).c_str(), fmt(r.decode_tps, 2).c_str(), tsv_escape(r.text).c_str());
    fflush(stdout);
    fprintf(stderr, "[%s %s p%d r%d] e2e %8.2f ms  ttft %8.2f ms  prefill %3s  gen %3s  decode %7s tok/s | %s\n",
            r.variant.c_str(), r.mode.c_str(), r.prompt_idx, r.rep, r.e2e, r.ttft, fmt_int(r.n_prefill).c_str(),
            fmt_int(r.n_gen).c_str(), fmt(r.decode_tps, 1).c_str(), tsv_escape(r.text).c_str());
}

void print_summary(const std::string & label, const std::vector<Row> & rows) {
    std::vector<double> e2e, ttft, dec;
    for (const auto & r : rows) {
        e2e.push_back(r.e2e);
        ttft.push_back(r.ttft);
        dec.push_back(r.decode_tps);
    }
    fprintf(stderr, "# %s: n=%zu  e2e median %.2f ms (min %.2f)  ttft median %.2f ms  decode median %.1f tok/s\n",
            label.c_str(), rows.size(), median(e2e), min_of(e2e), median(ttft), median(dec));
}

// ---------------------------------------------------------------------------------------------
// options

struct Options {
    std::string model, variant, mode = "full";
    std::string prompts_path = "bench/prompts.txt", system_path = "bench/system.txt";
    int reps = 3, rep_start = 0, warmup = -1, n_predict = 64, n_ctx = 512;
    std::vector<int> prompt_idx;                     // empty = all prompts
    std::vector<std::string> fa = {"on"};            // on | off | auto
    std::vector<int> n_batch = {512}, n_ubatch = {512}, threads = {1};
    std::vector<std::string> sampler = {"cpu"};      // cpu | backend
    bool header = true, verbose = false;
    std::string cli = "/opt/homebrew/bin/llama-completion";
    std::vector<std::string> cli_args;               // extra cold-mode CLI args
};

void usage() {
    fprintf(stderr,
        "usage: bench --model FILE.gguf [options]\n"
        "  --mode full|prefix|cold     (default full)\n"
        "  --variant NAME              label for the TSV variant column (default: from file name)\n"
        "  --prompts FILE --system FILE (default bench/prompts.txt, bench/system.txt)\n"
        "  --prompt-idx 0,3,5          subset of prompts (0-based, default all)\n"
        "  --reps N --rep-start N      timed passes over the prompts and first rep label (3, 0)\n"
        "  --warmup N                  untimed requests first (-1 = one pass over the prompts)\n"
        "  --n-predict N --n-ctx N     (64, 512)\n"
        "  --ignore-eos                always generate exactly n-predict tokens\n"
        "  --fa on|off|auto --n-batch N --n-ubatch N --threads N --sampler cpu|backend\n"
        "                              (on, 512, 512, 1, cpu); each takes a comma-separated list\n"
        "                              and the cartesian product is run, one context per combination\n"
        "  --cli PATH                  one-shot CLI for cold mode (default llama-completion)\n"
        "  --cli-arg ARG               extra CLI argument for cold mode (repeatable)\n"
        "  --no-header --verbose\n");
    exit(2);
}

Options parse_args(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for %s", a.c_str());
            return argv[++i];
        };
        if      (a == "--model")      o.model = next();
        else if (a == "--variant")    o.variant = next();
        else if (a == "--mode")       o.mode = next();
        else if (a == "--prompts")    o.prompts_path = next();
        else if (a == "--system")     o.system_path = next();
        else if (a == "--prompt-idx") o.prompt_idx = split_ints(next());
        else if (a == "--reps")       o.reps = std::stoi(next());
        else if (a == "--rep-start")  o.rep_start = std::stoi(next());
        else if (a == "--warmup")     o.warmup = std::stoi(next());
        else if (a == "--n-predict")  o.n_predict = std::stoi(next());
        else if (a == "--ignore-eos") g_ignore_eos = true;
        else if (a == "--n-ctx")      o.n_ctx = std::stoi(next());
        else if (a == "--fa")         o.fa = split(next(), ',');
        else if (a == "--n-batch")    o.n_batch = split_ints(next());
        else if (a == "--n-ubatch")   o.n_ubatch = split_ints(next());
        else if (a == "--threads")    o.threads = split_ints(next());
        else if (a == "--sampler")    o.sampler = split(next(), ',');
        else if (a == "--cli")        o.cli = next();
        else if (a == "--cli-arg")    o.cli_args.push_back(next());
        else if (a == "--no-header")  o.header = false;
        else if (a == "--verbose")    o.verbose = true;
        else if (a == "-h" || a == "--help") usage();
        else die("unknown argument %s (try --help)", a.c_str());
    }
    if (o.model.empty()) usage();
    if (o.mode != "full" && o.mode != "prefix" && o.mode != "cold") die("bad --mode %s", o.mode.c_str());
    if (o.variant.empty()) {
        // qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -> q4_k_m
        std::string base = o.model.substr(o.model.find_last_of('/') + 1);
        if (ends_with(base, ".gguf")) base.resize(base.size() - 5);
        const size_t dash = base.find_last_of('-');
        o.variant = dash == std::string::npos ? base : base.substr(dash + 1);
    }
    for (const auto & f : o.fa) {
        if (f != "on" && f != "off" && f != "auto") die("bad --fa %s", f.c_str());
    }
    for (const auto & s : o.sampler) {
        if (s != "cpu" && s != "backend") die("bad --sampler %s", s.c_str());
    }
    return o;
}

// ---------------------------------------------------------------------------------------------
// logging: keep libllama quiet unless --verbose; always pass through warnings/errors and the
// line that reports the resolved flash-attention setting.

bool g_verbose = false;

void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN ||
        (text && strstr(text, "flash_attn") && strstr(text, "->"))) {
        fputs(text, stderr);
    }
}

// ---------------------------------------------------------------------------------------------
// in-process runner (full / prefix)

struct Config {
    std::string fa, sampler;
    int n_batch, n_ubatch, threads;

    std::string label() const {
        return "fa=" + fa + ",nb=" + std::to_string(n_batch) + ",nub=" + std::to_string(n_ubatch) +
               ",t=" + std::to_string(threads) + ",s=" + sampler;
    }
};

enum llama_flash_attn_type fa_type(const std::string & s) {
    if (s == "on") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (s == "off") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

struct Runner {
    const llama_vocab * vocab = nullptr;
    llama_context * ctx = nullptr;
    llama_memory_t mem = nullptr;
    llama_sampler * smpl = nullptr;
    llama_batch batch{};
    llama_sampler_seq_config seq_cfg{};
    int n_batch = 512, n_predict = 64, n_prefix = 0;
    std::string system;
    std::vector<llama_token> toks;

    Runner(llama_model * model, const Options & o, const Config & c, const std::string & system_prompt)
        : vocab(llama_model_get_vocab(model)), n_batch(c.n_batch), n_predict(o.n_predict), system(system_prompt) {
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        smpl = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) o.n_ctx;
        cp.n_batch = (uint32_t) c.n_batch;
        cp.n_ubatch = (uint32_t) c.n_ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = c.threads;
        cp.n_threads_batch = c.threads;
        cp.flash_attn_type = fa_type(c.fa);
        cp.offload_kqv = true;
        cp.no_perf = true;
        if (c.sampler == "backend") {
            seq_cfg = {0, smpl};
            cp.samplers = &seq_cfg;
            cp.n_samplers = 1;
        }
        ctx = llama_init_from_model(model, cp);
        if (!ctx) die("llama_init_from_model failed (%s)", c.label().c_str());
        mem = llama_get_memory(ctx);
        batch = llama_batch_init(c.n_batch, 0, 1);
    }

    ~Runner() {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_sampler_free(smpl);
    }

    int tokenize(const std::string & s, std::vector<llama_token> & out) const {
        if (out.size() < s.size() + 16) out.resize(s.size() + 16);
        int n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(),
                               /*add_special=*/false, /*parse_special=*/true);
        if (n < 0) {
            out.resize((size_t) -n);
            n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(), false, true);
        }
        if (n <= 0) die("tokenize failed");
        return n;
    }

    // Byte-level pieces concatenated = the decoded text. Special tokens other than the stop token
    // are rendered, like the fp32 reference (tokenizers decode, skip_special_tokens=False).
    void append_piece(std::string & text, llama_token tok) const {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof buf, 0, /*special=*/true);
        if (n < 0) {
            std::string big((size_t) -n, '\0');
            n = llama_token_to_piece(vocab, tok, big.data(), (int32_t) big.size(), 0, true);
            text.append(big.data(), (size_t) std::max(n, 0));
            return;
        }
        text.append(buf, (size_t) n);
    }

    void set_token(int j, llama_token t, llama_pos pos, bool logits) {
        batch.token[j] = t;
        batch.pos[j] = pos;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = logits ? 1 : 0;
    }

    void prefill(const llama_token * t, int n, int pos0) {
        for (int i = 0; i < n; i += n_batch) {
            const int m = std::min(n_batch, n - i);
            batch.n_tokens = m;
            for (int j = 0; j < m; j++) set_token(j, t[i + j], pos0 + i + j, i + j == n - 1);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) die("llama_decode (prefill) failed: %d", rc);
        }
    }

    std::string prefix_text() const { return std::string(kPrefixHead) + system + kPrefixTail; }

    void setup_prefix() {
        const int n = tokenize(prefix_text(), toks);
        llama_memory_clear(mem, true);
        prefill(toks.data(), n, 0);
        llama_synchronize(ctx);
        n_prefix = n;
    }

    // Startup sanity check (untimed): tokenizing prefix and suffix separately must give exactly the
    // tokens of the whole templated prompt, otherwise prefix mode would change the model input.
    void check_split(const std::vector<std::string> & prompts) const {
        std::vector<llama_token> a, b, c;
        const int np = tokenize(prefix_text(), a);
        a.resize(np);
        for (size_t i = 0; i < prompts.size(); i++) {
            const int nf = tokenize(prefix_text() + prompts[i] + kSuffix, b);
            const int ns = tokenize(prompts[i] + kSuffix, c);
            bool ok = nf == np + ns;
            for (int j = 0; ok && j < nf; j++) ok = b[j] == (j < np ? a[j] : c[j - np]);
            if (!ok) fprintf(stderr, "bench: WARNING prompt %zu: prefix+suffix tokens differ from full tokens\n", i);
        }
    }

    Row run(const std::string & prompt, bool prefix_mode) {
        Row r;
        const double t0 = now_ms();
        std::string s;
        int pos = 0;
        if (prefix_mode) {
            s.reserve(prompt.size() + 40);
            s.append(prompt).append(kSuffix);
        } else {
            s.reserve(system.size() + prompt.size() + 96);
            s.append(kPrefixHead).append(system).append(kPrefixTail).append(prompt).append(kSuffix);
        }
        const int n = tokenize(s, toks);
        if (prefix_mode) {
            if (!llama_memory_seq_rm(mem, 0, n_prefix, -1)) die("llama_memory_seq_rm failed");
            pos = n_prefix;
        } else {
            llama_memory_clear(mem, false);
        }
        llama_sampler_reset(smpl);
        prefill(toks.data(), n, pos);
        pos += n;
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        const double t1 = now_ms();
        double t_last = t1;
        int n_gen = 1;
        std::string text;
        while (g_ignore_eos || (tok != kImEnd && tok != kEndOfText)) {
            if (n_gen >= n_predict) {
                append_piece(text, tok);
                break;
            }
            batch.n_tokens = 1;
            set_token(0, tok, pos++, true);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) die("llama_decode failed: %d", rc);
            append_piece(text, tok); // overlaps with the GPU work submitted above
            tok = llama_sampler_sample(smpl, ctx, -1);
            n_gen++;
            t_last = now_ms();
        }
        const double t2 = now_ms();
        r.e2e = t2 - t0;
        r.ttft = t1 - t0;
        r.n_prefill = n;
        r.n_gen = n_gen;
        r.decode_tps = n_gen > 1 ? (n_gen - 1) * 1000.0 / (t_last - t1) : NAN;
        r.text = std::move(text);
        return r;
    }
};

// ---------------------------------------------------------------------------------------------
// cold mode: spawn one-shot CLI processes

struct SpawnResult {
    double e2e = NAN, first_out = NAN;
    int status = 0;
    std::string out, err;
};

SpawnResult spawn_capture(const std::vector<std::string> & argv) {
    int po[2], pe[2];
    if (pipe(po) != 0 || pipe(pe) != 0) die("pipe: %s", strerror(errno));
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, po[1], 1);
    posix_spawn_file_actions_adddup2(&fa, pe[1], 2);
    posix_spawn_file_actions_addclose(&fa, po[0]);
    posix_spawn_file_actions_addclose(&fa, pe[0]);
    posix_spawn_file_actions_addclose(&fa, po[1]);
    posix_spawn_file_actions_addclose(&fa, pe[1]);
    std::vector<char *> args;
    for (const auto & a : argv) args.push_back(const_cast<char *>(a.c_str()));
    args.push_back(nullptr);

    SpawnResult r;
    pid_t pid = 0;
    const double t0 = now_ms();
    const int rc = posix_spawn(&pid, args[0], &fa, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) die("posix_spawn %s: %s", args[0], strerror(rc));
    close(po[1]);
    close(pe[1]);

    struct pollfd fds[2] = {{po[0], POLLIN, 0}, {pe[0], POLLIN, 0}};
    int open_fds = 2;
    char buf[1 << 16];
    while (open_fds > 0) {
        const int n = poll(fds, 2, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            die("poll: %s", strerror(errno));
        }
        for (int i = 0; i < 2; i++) {
            if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            const ssize_t k = read(fds[i].fd, buf, sizeof buf);
            if (k > 0) {
                if (i == 0) {
                    if (!std::isfinite(r.first_out)) r.first_out = now_ms() - t0;
                    r.out.append(buf, (size_t) k);
                } else {
                    r.err.append(buf, (size_t) k);
                }
            } else if (k == 0 || (errno != EINTR && errno != EAGAIN)) {
                close(fds[i].fd);
                fds[i].fd = -1;
                open_fds--;
            }
        }
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    r.e2e = now_ms() - t0;
    r.status = status;
    return r;
}

// "llama_perf_context_print: prompt eval time =      12.34 ms /    63 tokens (...)"
// "llama_perf_context_print:        eval time =     123.45 ms /    19 runs   (...)"
void parse_perf(const std::string & err, Row & r) {
    std::stringstream ss(err);
    std::string line;
    double eval_ms = NAN;
    int runs = -1;
    while (std::getline(ss, line)) {
        const size_t p = line.find("eval time =");
        if (p == std::string::npos) continue;
        double ms = 0;
        int cnt = 0;
        if (sscanf(line.c_str() + p, "eval time = %lf ms / %d", &ms, &cnt) != 2) continue;
        if (line.find("prompt eval time") != std::string::npos) {
            r.n_prefill = cnt;
        } else {
            eval_ms = ms;
            runs = cnt;
        }
    }
    if (runs >= 0) {
        r.n_gen = runs + 1; // the last sampled token (stop token or the n_predict-th) is never evaluated
        r.decode_tps = runs > 0 && eval_ms > 0 ? runs * 1000.0 / eval_ms : NAN;
    }
}

std::vector<std::string> cold_argv(const Options & o, const Config & c, const std::string & templated) {
    std::vector<std::string> a = {
        o.cli, "-m", o.model,
        "-ngl", "99", "-c", std::to_string(o.n_ctx), "-n", std::to_string(o.n_predict),
        "-fa", c.fa, "-b", std::to_string(c.n_batch), "-ub", std::to_string(c.n_ubatch),
        "-t", std::to_string(c.threads),
        "--temp", "0", "--samplers", "top_k", "--top-k", "1", // greedy: one candidate survives top_k
        "-fit", "off", "--no-warmup",                           // both only cost start-up time
        "-no-cnv", "--no-display-prompt", "--simple-io", "--no-escape", "--perf",
    };
    if (c.sampler == "backend") a.push_back("--backend-sampling");
    for (const auto & x : o.cli_args) a.push_back(x);
    a.push_back("-p");
    a.push_back(templated);
    return a;
}

Row run_cold(const Options & o, const Config & c, const std::string & templated) {
    const SpawnResult s = spawn_capture(cold_argv(o, c, templated));
    if (!WIFEXITED(s.status) || WEXITSTATUS(s.status) != 0) {
        fprintf(stderr, "%s\n", s.err.c_str());
        die("%s exited abnormally (status %d)", o.cli.c_str(), s.status);
    }
    Row r;
    r.e2e = s.e2e;
    r.ttft = s.first_out;
    std::string out = s.out;
    if (ends_with(out, "\n\n")) {
        out.resize(out.size() - 2);
    } else {
        fprintf(stderr, "bench: WARNING cold stdout lacks the trailing blank line; output kept verbatim\n");
    }
    if (ends_with(out, " [end of text]\n")) out.resize(out.size() - strlen(" [end of text]\n"));
    r.text = out;
    parse_perf(s.err, r);
    return r;
}

std::string templated_prompt(const std::string & system, const std::string & prompt) {
    return std::string(kPrefixHead) + system + kPrefixTail + prompt + kSuffix;
}

} // namespace

int main(int argc, char ** argv) {
    const Options o = parse_args(argc, argv);
    g_verbose = o.verbose;

    const std::vector<std::string> all_prompts = read_prompts(o.prompts_path);
    std::string system = read_file(o.system_path);
    while (!system.empty() && (system.back() == '\n' || system.back() == '\r')) system.pop_back();

    std::vector<int> idx = o.prompt_idx;
    if (idx.empty()) {
        for (int i = 0; i < (int) all_prompts.size(); i++) idx.push_back(i);
    }
    for (int i : idx) {
        if (i < 0 || i >= (int) all_prompts.size()) die("prompt index %d out of range", i);
    }
    const int n_warmup = o.warmup < 0 ? (int) idx.size() : o.warmup;

    std::vector<Config> configs;
    for (const auto & fa : o.fa)
        for (int nb : o.n_batch)
            for (int nub : o.n_ubatch)
                for (int t : o.threads)
                    for (const auto & s : o.sampler) {
                        if (nub > nb) continue;
                        configs.push_back({fa, s, nb, nub, t});
                    }
    if (configs.empty()) die("no valid configuration (n_ubatch must be <= n_batch)");

    if (o.header) print_header();

    if (o.mode == "cold") {
        for (const auto & c : configs) {
            const std::string variant = configs.size() > 1 ? o.variant + "/" + c.label() : o.variant;
            fprintf(stderr, "# llama.cpp %s cold: %s\n", variant.c_str(), o.cli.c_str());
            // one untimed spawn so the model file and the Metal shader cache are warm
            for (int w = 0; w < std::min(n_warmup, 1); w++) run_cold(o, c, templated_prompt(system, all_prompts[idx[0]]));
            std::vector<Row> rows;
            for (int rep = 0; rep < o.reps; rep++) {
                for (int i : idx) {
                    Row r = run_cold(o, c, templated_prompt(system, all_prompts[i]));
                    r.variant = variant;
                    r.mode = "cold";
                    r.prompt_idx = i;
                    r.rep = o.rep_start + rep;
                    print_row(r);
                    rows.push_back(r);
                }
            }
            print_summary("llama.cpp " + variant + " cold", rows);
        }
        return 0;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    bool have_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (o.verbose) fprintf(stderr, "device %zu: %s (%s)\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        if (strstr(ggml_backend_dev_name(d), "MTL") || strstr(ggml_backend_dev_name(d), "Metal")) have_gpu = true;
    }
    if (!have_gpu) die("no Metal device registered (ggml backends not found?)");

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999; // all layers (and the output layer) on Metal
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) die("cannot load %s", o.model.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (strcmp(llama_vocab_get_text(vocab, kImEnd), "<|im_end|>") != 0 ||
        strcmp(llama_vocab_get_text(vocab, kEndOfText), "<|endoftext|>") != 0) {
        die("unexpected vocabulary: stop token ids do not match <|im_end|>/<|endoftext|>");
    }
    const bool prefix_mode = o.mode == "prefix";

    for (const auto & c : configs) {
        const std::string variant = configs.size() > 1 ? o.variant + "/" + c.label() : o.variant;
        fprintf(stderr, "# llama.cpp %s %s: %s, n_ctx %d\n", variant.c_str(), o.mode.c_str(), c.label().c_str(), o.n_ctx);
        Runner runner(model, o, c, system);
        if (prefix_mode) {
            runner.check_split(all_prompts);
            runner.setup_prefix();
            fprintf(stderr, "# prefix: %d tokens cached\n", runner.n_prefix);
        }
        for (int w = 0; w < n_warmup; w++) runner.run(all_prompts[idx[w % idx.size()]], prefix_mode);
        std::vector<Row> rows;
        for (int rep = 0; rep < o.reps; rep++) {
            for (int i : idx) {
                Row r = runner.run(all_prompts[i], prefix_mode);
                r.variant = variant;
                r.mode = o.mode;
                r.prompt_idx = i;
                r.rep = o.rep_start + rep;
                print_row(r);
                rows.push_back(r);
            }
        }
        print_summary("llama.cpp " + variant + " " + o.mode, rows);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
