"""Task 3 answer normalization and equivalence (DECISIONS.md T3-3 to T3-6).

MATH: math-verify equivalence on the supplied extracted answers, each parsed as `$<answer>$`.
MathVista: the official normalization from lupantech/MathVista `evaluation/calculate_score.py`
(commit 9d65741c1f0a136989c2589e6e6e57e687261c07): `normalize_extracted_answer` with its default
`ignore_empty_extractions=False`, `get_most_similar` and `safe_equal`, ported below. Multiple-choice
answers are written as option letters (manual). Missing extractions and extractions the type rules
cannot normalize become NO_ANSWER (manual; T3-4), always graded wrong.
"""

import re

from Levenshtein import distance

NO_ANSWER = "NO_ANSWER"


# --------------------------------------------------------------------------- MathVista (official port)

def get_most_similar(prediction, choices):
    """Use the Levenshtein distance (or edit distance) to determine which of the choices is most
    similar to the given prediction (official)."""
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
    """Official `normalize_extracted_answer` with ignore_empty_extractions=False."""
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def safe_equal(prediction, answer):
    """Official: equality that never raises."""
    try:
        return prediction == answer
    except Exception:
        return False


def mathvista_key(extracted, meta):
    """Group key for one rollout: an option letter, a normalized free-form string, or NO_ANSWER."""
    if extracted is None:
        return NO_ANSWER
    choices = list(meta["choices"]) if meta["question_type"] == "multi_choice" else None
    norm = normalize_extracted_answer(extracted, choices, meta["question_type"], meta["answer_type"],
                                      meta["precision"])
    if norm is None:
        return NO_ANSWER
    if meta["question_type"] == "multi_choice":
        return chr(ord("A") + choices.index(norm))
    return norm


def mathvista_gold_key(meta):
    """The metadata answer in the same representation (a letter for multiple choice)."""
    if meta["question_type"] == "multi_choice":
        return chr(ord("A") + list(meta["choices"]).index(meta["answer"]))
    return meta["answer"]


# --------------------------------------------------------------------------- MATH (math-verify)

def math_parse(answer):
    from math_verify import parse
    return parse(f"${answer}$")


def math_equal(gold_parsed, pred_parsed):
    from math_verify import verify
    return bool(verify(gold_parsed, pred_parsed))
