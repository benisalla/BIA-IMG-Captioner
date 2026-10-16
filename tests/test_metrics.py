import math

import pytest

from bia_captioner.metrics import corpus_bleu, exact_match, tokenize


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("A dog's ball, RED.") == ["a", "dog's", "ball", "red"]


def test_perfect_match_scores_one() -> None:
    scores = corpus_bleu(["a man rides a red bike"], [["a man rides a red bike", "other"]])
    assert scores == pytest.approx({"bleu1": 1.0, "bleu2": 1.0, "bleu3": 1.0, "bleu4": 1.0})


def test_no_overlap_scores_zero() -> None:
    scores = corpus_bleu(["cat cat cat cat"], [["a man rides a bike"]])
    assert all(value == 0.0 for value in scores.values())


def test_clipping_and_brevity_penalty() -> None:
    # "the the the" vs "the cat": unigram precision clipped to 1/3; hyp longer -> no penalty.
    assert corpus_bleu(["the the the"], [["the cat"]])["bleu1"] == pytest.approx(1 / 3)
    # "a dog" vs a 4-word reference: precision 1, brevity penalty exp(1 - 4/2).
    assert corpus_bleu(["a dog"], [["a dog runs fast"]])["bleu1"] == pytest.approx(math.exp(-1))


def test_known_bleu4_value() -> None:
    # 6-word hypothesis, closest reference also 6 words -> no brevity penalty.
    # matches: 1-gram 5/6, 2-gram 3/5, 3-gram 2/4, 4-gram 1/3.
    scores = corpus_bleu(["a man is riding a horse"], [["a man is riding the horse"]])
    expected = (5 / 6 * 3 / 5 * 2 / 4 * 1 / 3) ** 0.25
    assert scores["bleu4"] == pytest.approx(expected)


def test_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="one list"):
        corpus_bleu(["a"], [])


def test_exact_match_ignores_case_and_punctuation() -> None:
    hyps = ["A big red circle.", "a small blue square", "a big red square"]
    refs = [["a big red circle"], ["other", "A small blue square!"], ["a big red circle"]]
    assert exact_match(hyps, refs) == pytest.approx(2 / 3)
    assert exact_match([], []) == 0.0
