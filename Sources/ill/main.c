// ill: the instantllm client. Sends the prompt to the resident `instantllm serve` daemon over a
// Unix socket and streams the generated command to stdout. Starts the daemon if it is not running.
// Plain C so process start costs ~1 ms (no Swift runtime, no Metal).
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <spawn.h>

extern char **environ;

static void sock_path(char *out, size_t n) {
    const char *env = getenv("INSTANTLLM_SOCKET");
    if (env && *env) { snprintf(out, n, "%s", env); return; }
    const char *tmp = getenv("TMPDIR");
    snprintf(out, n, "%s/instantllm-%d.sock", tmp && *tmp ? tmp : "/tmp", (int)getuid());
}

static int try_connect(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un a = {0};
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof a.sun_path, "%s", path);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) == 0) return fd;
    close(fd);
    return -1;
}

static int write_all(int fd, const void *p, size_t n) {
    const char *c = p;
    while (n) {
        ssize_t w = write(fd, c, n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        c += w; n -= (size_t)w;
    }
    return 0;
}

// Starts `instantllm serve --detach` from the directory this binary lives in.
static void spawn_daemon(void) {
    char self[PATH_MAX], real[PATH_MAX];
    uint32_t sz = sizeof self;
    if (_NSGetExecutablePath(self, &sz) != 0 || !realpath(self, real)) return;
    char *slash = strrchr(real, '/');
    if (!slash) return;
    strcpy(slash + 1, "instantllm");
    char *argv[] = {real, "serve", "--detach", NULL};
    pid_t pid;
    posix_spawn(&pid, real, NULL, NULL, argv, environ);
}

int main(int argc, char **argv) {
    int max_new = 64;
    const char *system_prompt = NULL;
    char prompt[65536];
    size_t plen = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) { max_new = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "-s") && i + 1 < argc) { system_prompt = argv[++i]; continue; }
        size_t l = strlen(argv[i]);
        if (plen + l + 2 >= sizeof prompt) break;
        if (plen) prompt[plen++] = ' ';
        memcpy(prompt + plen, argv[i], l);
        plen += l;
    }
    if (plen == 0) {
        fprintf(stderr, "usage: ill [-n max_tokens] [-s system_prompt] <request...>\n");
        return 2;
    }
    char path[PATH_MAX];
    sock_path(path, sizeof path);
    int fd = try_connect(path);
    if (fd < 0) {
        spawn_daemon();
        for (int t = 0; t < 400 && fd < 0; t++) {  // up to ~10 s for the first model load
            struct timespec ts = {0, 25 * 1000 * 1000};
            nanosleep(&ts, NULL);
            fd = try_connect(path);
        }
        if (fd < 0) { fprintf(stderr, "ill: cannot reach instantllm daemon at %s\n", path); return 1; }
    }
    uint32_t slen = system_prompt ? (uint32_t)strlen(system_prompt) : 0;
    uint32_t hdr[4] = {0x31514C49u /* "ILQ1" */, (uint32_t)max_new, slen, (uint32_t)plen};
    if (write_all(fd, hdr, sizeof hdr) || (slen && write_all(fd, system_prompt, slen)) || write_all(fd, prompt, plen)) {
        fprintf(stderr, "ill: write failed\n");
        return 1;
    }
    char buf[4096];
    for (;;) {
        ssize_t r = read(fd, buf, sizeof buf);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        write_all(1, buf, (size_t)r);
    }
    write_all(1, "\n", 1);
    return 0;
}
