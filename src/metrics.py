"""
GEC Evaluation Metrics

Implements standard metrics for Grammatical Error Correction:
- F0.5 (precision-weighted)
- GLEU (GEC-specific BLEU variant)
- Exact Match Accuracy
- Edit Distance metrics
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import Levenshtein  # pip install python-Levenshtein


def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: Model predictions
        references: Gold corrections

    Returns:
        Accuracy (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(predictions)


def compute_gleu(predictions: List[str],
                 references: List[str],
                 sources: List[str] = None,
                 n: int = 4) -> float:
    """
    Compute GLEU (Generalized Language Evaluation Understanding).

    GLEU is a variant of BLEU designed for GEC that:
    1. Operates at sentence level (not corpus level)
    2. Penalizes n-grams that appear in source but not reference

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences (optional, for source penalty)
        n: Maximum n-gram order (default: 4)

    Returns:
        GLEU score (0-100)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    if sources is not None and len(sources) != len(predictions):
        raise ValueError("sources must have same length as predictions")

    gleu_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        src_tokens = sources[i].split() if sources else []

        # Compute precision for each n-gram order
        precisions = []
        for k in range(1, n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, k)
            ref_ngrams = _get_ngrams(ref_tokens, k)

            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue

            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += 1

            # GLEU modification: penalize copying from source
            if sources:
                src_ngrams = _get_ngrams(src_tokens, k)
                for ngram in pred_ngrams:
                    if ngram in src_ngrams and ngram not in ref_ngrams:
                        matches -= 0.5  # Penalty for unchanged source n-grams

            precision = max(0, matches) / len(pred_ngrams)
            precisions.append(precision)

        # Brevity penalty (like BLEU)
        bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            geo_mean = 0.0

        gleu_scores.append(bp * geo_mean)

    # GLEU is averaged at sentence level
    return np.mean(gleu_scores) * 100


def compute_f05_simple(predictions: List[str],
                       references: List[str],
                       sources: List[str]) -> Dict[str, float]:
    """
    Compute simple F0.5 score based on token-level edits.

    This is a simplified version that computes F0.5 based on:
    - TP: Tokens that were correctly changed from source to match reference
    - FP: Tokens that were changed but don't match reference
    - FN: Tokens in reference that weren't predicted

    For proper GEC evaluation, use ERRANT (compute_f05_errant).
    This is a reasonable approximation for quick experiments.

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences

    Returns:
        Dict with precision, recall, f05
    """
    if len(predictions) != len(references) != len(sources):
        raise ValueError("All inputs must have same length")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, ref, src in zip(predictions, references, sources):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        src_tokens = src.split()

        # Align tokens (simple word-level alignment)
        # TP: Correctly changed tokens
        # FP: Incorrectly changed tokens
        # FN: Missed corrections

        max_len = max(len(pred_tokens), len(ref_tokens), len(src_tokens))

        # Pad to same length for simple alignment
        pred_tokens += [''] * (max_len - len(pred_tokens))
        ref_tokens += [''] * (max_len - len(ref_tokens))
        src_tokens += [''] * (max_len - len(src_tokens))

        for p, r, s in zip(pred_tokens, ref_tokens, src_tokens):
            # If reference changed from source
            if r != s:
                if p == r:
                    total_tp += 1  # Correct correction
                else:
                    total_fn += 1  # Missed correction

            # If prediction changed from source
            if p != s and p != r:
                total_fp += 1  # Incorrect change

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # F0.5 gives more weight to precision
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f05': f05 * 100
    }


def compute_edit_distance_metrics(predictions: List[str],
                                   references: List[str]) -> Dict[str, float]:
    """
    Compute edit distance based metrics.

    Args:
        predictions: Model predictions
        references: Gold corrections

    Returns:
        Dict with various edit distance metrics
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    char_distances = []
    word_distances = []

    for pred, ref in zip(predictions, references):
        # Character-level Levenshtein distance
        char_dist = Levenshtein.distance(pred, ref)
        char_distances.append(char_dist)

        # Word-level edit distance
        pred_words = pred.split()
        ref_words = ref.split()
        word_dist = Levenshtein.distance(' '.join(pred_words), ' '.join(ref_words))
        word_distances.append(word_dist)

    return {
        'avg_char_edit_distance': np.mean(char_distances),
        'avg_word_edit_distance': np.mean(word_distances),
        'median_char_edit_distance': np.median(char_distances),
        'median_word_edit_distance': np.median(word_distances)
    }


def compute_all_metrics(predictions: List[str],
                       references: List[str],
                       sources: List[str]) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Exact match
    metrics['exact_match'] = exact_match_accuracy(predictions, references) * 100

    # GLEU
    metrics['gleu'] = compute_gleu(predictions, references, sources)

    # F0.5 (simple version)
    f05_metrics = compute_f05_simple(predictions, references, sources)
    metrics.update(f05_metrics)

    # Edit distance
    edit_metrics = compute_edit_distance_metrics(predictions, references)
    metrics.update(edit_metrics)

    return metrics


# Helper functions

def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    """Compute BLEU-style brevity penalty."""
    if pred_len >= ref_len:
        return 1.0
    else:
        return np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0


# Optional: ERRANT-based F0.5 (requires errant package)

def compute_f05_errant(predictions: List[str],
                       references: List[str],
                       sources: List[str],
                       lang: str = 'en') -> Dict[str, float]:
    """
    Compute F0.5 using ERRANT (Error Annotation Toolkit).

    This is the gold standard for GEC evaluation.
    Requires: pip install errant

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences
        lang: Language code ('en' for English, 'fi' for Finnish if available)

    Returns:
        Dict with precision, recall, f05
    """
    try:
        import errant
    except ImportError:
        raise ImportError(
            "ERRANT not installed. Install with: pip install errant\n"
            "Or use compute_f05_simple() for a simpler approximation."
        )

    # Load ERRANT annotator
    try:
        annotator = errant.load(lang)
    except:
        # Fallback to English if language not available
        print(f"Warning: {lang} not available in ERRANT, falling back to English")
        annotator = errant.load('en')

    all_edits_pred = []
    all_edits_gold = []

    for pred, ref, src in zip(predictions, references, sources):
        # Parse sentences
        src_parsed = annotator.parse(src)
        pred_parsed = annotator.parse(pred)
        ref_parsed = annotator.parse(ref)

        # Extract edits
        pred_edits = annotator.annotate(src_parsed, pred_parsed)
        gold_edits = annotator.annotate(src_parsed, ref_parsed)

        all_edits_pred.append(pred_edits)
        all_edits_gold.append(gold_edits)

    # Compute metrics
    tp = 0
    fp = 0
    fn = 0

    for pred_edits, gold_edits in zip(all_edits_pred, all_edits_gold):
        # Count true positives
        for pred_edit in pred_edits:
            if pred_edit in gold_edits:
                tp += 1
            else:
                fp += 1

        # Count false negatives
        for gold_edit in gold_edits:
            if gold_edit not in pred_edits:
                fn += 1

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f05': f05 * 100
    }
