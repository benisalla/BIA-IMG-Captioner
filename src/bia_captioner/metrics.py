"""Caption quality metrics: corpus BLEU-1..4 (Papineni et al., 2002).

Pure Python, no extra dependency. Matches `nltk.translate.bleu_score.corpus_bleu`
without smoothing: clipped n-gram precisions are summed over the whole corpus,
then combined with a brevity penalty that uses, for each hypothesis, the
reference length closest to it.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lower-case words without punctuation: "A dog, running." -> ["a", "dog", "running"]."""
    return _WORD.findall(text.lower())


def _ngrams(words: Sequence[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def corpus_bleu(
    hypotheses: Sequence[str], references: Sequence[Sequence[str]], max_n: int = 4
) -> dict[str, float]:
    """BLEU-1 .. BLEU-max_n of generated captions against their references.

    hypotheses: one generated caption per image.
    references: for each image, all of its human captions.
    Returns {"bleu1": ..., "bleu4": ...} with values in [0, 1].
    """
    if len(hypotheses) != len(references):
        raise ValueError("need exactly one list of references per hypothesis")
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    for hypothesis, refs in zip(hypotheses, references):
        hyp = tokenize(hypothesis)
        ref_tokens = [tokenize(r) for r in refs]
        hyp_len += len(hyp)
        # Closest reference length; ties go to the shorter one.
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in ref_tokens)[1]
        for n in range(1, max_n + 1):
            counts = _ngrams(hyp, n)
            # Clip: an n-gram counts at most as often as in the reference that has it most.
            max_ref = Counter()
            for r in ref_tokens:
                max_ref |= _ngrams(r, n)
            matches[n - 1] += sum(min(c, max_ref[g]) for g, c in counts.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)

    if hyp_len == 0:
        return {f"bleu{n}": 0.0 for n in range(1, max_n + 1)}
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    scores = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        if matches[n - 1] == 0 or totals[n - 1] == 0:
            # One empty precision makes this and every higher-order BLEU zero.
            for m in range(n, max_n + 1):
                scores[f"bleu{m}"] = 0.0
            break
        log_sum += math.log(matches[n - 1] / totals[n - 1])
        scores[f"bleu{n}"] = brevity * math.exp(log_sum / n)
    return scores


__all__ = ["corpus_bleu", "tokenize"]
