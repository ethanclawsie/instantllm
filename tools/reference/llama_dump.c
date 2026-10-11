// llama_dump.c: evaluate the golden prompts with the llama.cpp C API (libllama) and dump
// last-prompt-position logits, greedy tokens, per-step top-5 and teacher-forced argmax.
// Driven by run_llamacpp.py, which builds it, writes the job file and converts the output
// to the engine layout read by compare.py.
//
// Build:
//   clang -O2 -std=c11 -I/opt/homebrew/opt/llama.cpp/include -I/opt/homebrew/opt/ggml/include llama_dump.c \
//     -L/opt/homebrew/opt/llama.cpp/lib -lllama -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -o llama_dump
// Usage:
//   llama_dump MODEL.gguf JOBS.txt OUTDIR [n_gpu_layers=99] [max_new=64]
// JOBS.txt, one line per prompt (decimal integers separated by spaces):
//   <index> <n_prompt> <prompt ids...> <n_gold> <golden ids incl. stop token...>
// If OUTDIR/pNN.prompt.txt exists it is tokenized (add_special=true, parse_special=true)
// so the llama.cpp tokenizer can be checked against the golden input ids.
// Output per prompt: OUTDIR/pNN.logits.bin (float32, n_vocab) and OUTDIR/pNN.txt with lines
//   n_vocab N | tok <ids> | out <ids> | stop <id|-1> | forced <ids> | top5 <id:logit x5> (one per step)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

#define MAX_TOK 1024

static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static int argmax(const float *x, int n) {  // first maximum = lowest id on ties
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static void top5(const float *x, int n, int *ids) {
    for (int k = 0; k < 5; k++) ids[k] = -1;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 5; k++) {
            if (ids[k] < 0 || x[i] > x[ids[k]]) {
                for (int m = 4; m > k; m--) ids[m] = ids[m - 1];
                ids[k] = i;
                break;
            }
        }
    }
}

static void fill_batch(struct llama_batch *b, const llama_token *toks, int n, int pos0, int logits_from) {
    b->n_tokens = n;
    for (int i = 0; i < n; i++) {
        b->token[i] = toks[i];
        b->pos[i] = pos0 + i;
        b->n_seq_id[i] = 1;
        b->seq_id[i][0] = 0;
        b->logits[i] = i >= logits_from;
    }
}

static int is_stop(int t) { return t == 151645 || t == 151643; }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf JOBS.txt OUTDIR [n_gpu_layers=99] [max_new=64]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1], *jobs_path = argv[2], *outdir = argv[3];
    int ngl = argc > 4 ? atoi(argv[4]) : 99;
    int max_new = argc > 5 ? atoi(argv[5]) : 64;

    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.no_perf = true;  // other settings (flash attention auto, f16 KV cache) stay at defaults
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    struct llama_batch batch = llama_batch_init(512, 0, 1);
    llama_memory_t mem = llama_get_memory(ctx);

    FILE *jf = fopen(jobs_path, "r");
    if (!jf) { perror(jobs_path); return 1; }
    static llama_token prompt[MAX_TOK], gold[MAX_TOK], seq[MAX_TOK], out[MAX_TOK], forced[MAX_TOK];
    static int steps_top5[MAX_TOK][5];
    static float steps_top5_val[MAX_TOK][5];
    int index, n_prompt, n_gold;
    while (fscanf(jf, "%d %d", &index, &n_prompt) == 2) {
        if (n_prompt <= 0 || n_prompt >= MAX_TOK) { fprintf(stderr, "bad n_prompt\n"); return 1; }
        for (int i = 0; i < n_prompt; i++) if (fscanf(jf, "%d", &prompt[i]) != 1) return 1;
        if (fscanf(jf, "%d", &n_gold) != 1 || n_gold <= 0 || n_gold > max_new) { fprintf(stderr, "bad n_gold\n"); return 1; }
        for (int i = 0; i < n_gold; i++) if (fscanf(jf, "%d", &gold[i]) != 1) return 1;

        char path[4096];
        snprintf(path, sizeof path, "%s/p%02d.txt", outdir, index);
        FILE *tf = fopen(path, "w");
        if (!tf) { perror(path); return 1; }
        fprintf(tf, "n_vocab %d\n", n_vocab);

        // tokenizer check
        snprintf(path, sizeof path, "%s/p%02d.prompt.txt", outdir, index);
        FILE *pf = fopen(path, "rb");
        if (pf) {
            static char text[1 << 16];
            size_t len = fread(text, 1, sizeof text - 1, pf);
            fclose(pf);
            static llama_token toks[MAX_TOK];
            int nt = llama_tokenize(vocab, text, (int)len, toks, MAX_TOK, true, true);
            fprintf(tf, "tok");
            for (int i = 0; i < nt; i++) fprintf(tf, " %d", toks[i]);
            fprintf(tf, "\n");
        }

        // prefill + greedy decode
        llama_memory_clear(mem, true);
        fill_batch(&batch, prompt, n_prompt, 0, n_prompt - 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        const float *lg = llama_get_logits_ith(ctx, -1);
        snprintf(path, sizeof path, "%s/p%02d.logits.bin", outdir, index);
        FILE *lf = fopen(path, "wb");
        if (!lf) { perror(path); return 1; }
        fwrite(lg, sizeof(float), (size_t)n_vocab, lf);
        fclose(lf);
        int n_out = 0, stop = -1, n_steps = 0;
        for (int step = 0; step < max_new; step++) {
            int t = argmax(lg, n_vocab);
            top5(lg, n_vocab, steps_top5[n_steps]);
            for (int k = 0; k < 5; k++) steps_top5_val[n_steps][k] = lg[steps_top5[n_steps][k]];
            n_steps++;
            if (is_stop(t)) { stop = t; break; }
            out[n_out++] = t;
            if (n_out == max_new) break;
            llama_token one = t;
            fill_batch(&batch, &one, 1, n_prompt + n_out - 1, 0);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            lg = llama_get_logits_ith(ctx, -1);
        }

        // teacher forcing: prompt + golden tokens (minus the last) in one batch
        llama_memory_clear(mem, true);
        int n_seq = n_prompt + n_gold - 1;
        memcpy(seq, prompt, sizeof(llama_token) * n_prompt);
        memcpy(seq + n_prompt, gold, sizeof(llama_token) * (n_gold - 1));
        fill_batch(&batch, seq, n_seq, 0, n_prompt - 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int j = 0; j < n_gold; j++) forced[j] = argmax(llama_get_logits_ith(ctx, n_prompt - 1 + j), n_vocab);

        fprintf(tf, "out");
        for (int i = 0; i < n_out; i++) fprintf(tf, " %d", out[i]);
        fprintf(tf, "\nstop %d\nforced", stop);
        for (int j = 0; j < n_gold; j++) fprintf(tf, " %d", forced[j]);
        fprintf(tf, "\n");
        for (int s = 0; s < n_steps; s++) {
            fprintf(tf, "top5");
            for (int k = 0; k < 5; k++) fprintf(tf, " %d:%.9g", steps_top5[s][k], steps_top5_val[s][k]);
            fprintf(tf, "\n");
        }
        fclose(tf);
        fprintf(stderr, "[%2d] n_prompt=%d out=%d stop=%d\n", index, n_prompt, n_out, stop);
    }
    fclose(jf);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
