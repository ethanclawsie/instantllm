"""Edited neighbours: adapt a similar training request's output to the current request by
substituting the words that differ between the two prompts (e.g. "extract the audio from
video.mp4 as mp3" -> "... lecture.mp4 as wav": video->lecture, mp3->wav in the neighbour's
command). Everything here is ASCII-only and exactly specified so a Swift port can match it.

  words(text)          maximal runs of [A-Za-z0-9_.\\-/~] (original case)
  substitutions(b, a)  word-level LCS between neighbour prompt words b and current words a
                       (case-insensitive); every gap between consecutive LCS matches (and the
                       tail) whose two sides have equal length yields pairwise substitutions
                       b_word -> a_word for differing words (first mapping of a word wins)
  apply(text, subs)    for src in subs sorted by (-len, src): replace every occurrence of src
                       in text that is not preceded or followed by [A-Za-z0-9_], scanning left to
                       right without overlaps
"""
import re

_EW = re.compile(r"[A-Za-z0-9_.\-/~]+")


def words(text):
    return _EW.findall(text)


def _lcs_pairs(b, a):
    n, m = len(b), len(a)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if b[i] == a[j] else max(L[i + 1][j], L[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if b[i] == a[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def substitutions(b_orig, a_orig):
    b = [w.lower() for w in b_orig]
    a = [w.lower() for w in a_orig]
    subs = {}
    pi = pj = 0
    for i, j in _lcs_pairs(b, a) + [(len(b), len(a))]:
        if i - pi == j - pj and i > pi:
            for k in range(i - pi):
                if b[pi + k] != a[pj + k] and b_orig[pi + k] not in subs:
                    subs[b_orig[pi + k]] = a_orig[pj + k]
        pi, pj = i + 1, j + 1
    return subs


def _wordch(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9") or c == "_"


def apply(text, subs):
    for src in sorted(subs, key=lambda s: (-len(s), s)):
        dst = subs[src]
        out = []
        i = 0
        while True:
            k = text.find(src, i)
            if k < 0:
                out.append(text[i:])
                break
            e = k + len(src)
            if (k > 0 and _wordch(text[k - 1])) or (e < len(text) and _wordch(text[e])):
                out.append(text[i:k + 1])
                i = k + 1
                continue
            out.append(text[i:k])
            out.append(dst)
            i = e
        text = "".join(out)
    return text


def edited_sequence(tokenizer, cur_prompt, nb_prompt, nb_y, stops=(151645, 151643)):
    """The neighbour's output ids with the prompt substitutions applied, or None when no word
    differs. nb_y may end with a stop token, which is kept."""
    subs = substitutions(words(nb_prompt), words(cur_prompt))
    if not subs:
        return None
    stop = [nb_y[-1]] if nb_y and nb_y[-1] in stops else []
    body = nb_y[:-1] if stop else nb_y
    text = tokenizer.decode(body)
    new = apply(text, subs)
    if new == text:
        return None
    return tokenizer.encode(new, add_special_tokens=False).ids + stop
