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

    WHY F0.5 INSTEAD OF F1?
    - F0.5 weights precision 2.5x more than recall
    - In GEC, false positives (wrong corrections) are worse than false negatives (missed errors)
    - Better to leave text unchanged than introduce new errors
    - Standard metric for GEC evaluation (used in CoNLL, BEA shared tasks)

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
        'correction_precision': precision * 100,
        'correction_recall': recall * 100,
        'correction_f05': f05 * 100
    }


def compute_edit_distance_metrics(predictions: List[str],
                                   references: List[str]) -> Dict[str, float]:
    """
    Compute character-level edit distance metrics.

    For Finnish GEC, character-level is more appropriate than word-level
    due to rich morphology (case, number, person inflections).

    Args:
        predictions: Model predictions
        references: Gold corrections

    Returns:
        Dict with character edit distance metrics
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    char_distances = []
    normalized_distances = []

    for pred, ref in zip(predictions, references):
        # Character-level Levenshtein distance
        char_dist = Levenshtein.distance(pred, ref)
        char_distances.append(char_dist)

        # Normalized by reference length (percentage of characters that differ)
        normalized_dist = (char_dist / len(ref) * 100) if len(ref) > 0 else 0
        normalized_distances.append(normalized_dist)

    return {
        'correction_avg_char_edit_distance': np.mean(char_distances),
        'correction_median_char_edit_distance': np.median(char_distances),
        'correction_avg_normalized_edit_distance': np.mean(normalized_distances)  # As percentage
    }


def compute_bleu(predictions: List[str],
                 references: List[str],
                 max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU score.

    BLEU vs GLEU:
    - BLEU: Corpus-level metric (traditional MT evaluation)
    - GLEU: Sentence-level metric with source penalty (designed for GEC)

    Args:
        predictions: Model predictions
        references: Gold corrections
        max_n: Maximum n-gram order (default: 4)

    Returns:
        BLEU score (0-100)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    # Collect all n-gram counts across corpus
    total_matches = {n: 0 for n in range(1, max_n + 1)}
    total_possible = {n: 0 for n in range(1, max_n + 1)}
    total_pred_len = 0
    total_ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(_get_ngrams(ref_tokens, n))

            # Count matches (clipped by reference counts)
            for ngram, count in pred_ngrams.items():
                total_matches[n] += min(count, ref_ngrams.get(ngram, 0))
            total_possible[n] += len(_get_ngrams(pred_tokens, n))

    # Compute precisions for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        if total_possible[n] > 0:
            precisions.append(total_matches[n] / total_possible[n])
        else:
            precisions.append(0.0)

    # Brevity penalty
    if total_pred_len >= total_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - total_ref_len / total_pred_len) if total_pred_len > 0 else 0.0

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    return bp * geo_mean * 100


def compute_ged_metrics(predictions: List[str],
                        references: List[str],
                        sources: List[str]) -> Dict[str, float]:
    """
    Compute Grammatical Error Detection (GED) metrics.

    Measures whether the model can DETECT errors (regardless of correction quality).
    Not penalized by single-reference problems - only checks if errors were spotted.

    Token-level: Did model change tokens that actually had errors?
    Sentence-level: Did model correctly classify sentence as needing corrections?

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences

    Returns:
        Dict with GED metrics
    """
    # Token-level detection
    total_det_tp = 0  # Detected a real error position
    total_det_fp = 0  # Changed a correct position
    total_det_fn = 0  # Missed an error position

    # Sentence-level detection
    sent_det_tp = 0   # Correctly detected error sentence
    sent_det_fp = 0   # Wrongly flagged clean sentence
    sent_det_fn = 0   # Missed error sentence
    sent_tn = 0       # Correctly left clean sentence unchanged

    for pred, ref, src in zip(predictions, references, sources):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        src_tokens = src.split()

        # Pad to same length
        max_len = max(len(pred_tokens), len(ref_tokens), len(src_tokens))
        pred_tokens += [''] * (max_len - len(pred_tokens))
        ref_tokens += [''] * (max_len - len(ref_tokens))
        src_tokens += [''] * (max_len - len(src_tokens))

        # Token-level detection
        has_error = (src != ref)  # Does sentence have errors?
        made_change = (src != pred)  # Did model make changes?

        for p, r, s in zip(pred_tokens, ref_tokens, src_tokens):
            error_here = (s != r)  # Error at this position?
            detected_here = (s != p)  # Model changed this position?

            if detected_here and error_here:
                total_det_tp += 1
            elif detected_here and not error_here:
                total_det_fp += 1
            elif not detected_here and error_here:
                total_det_fn += 1

        # Sentence-level detection
        if has_error and made_change:
            sent_det_tp += 1
        elif not has_error and made_change:
            sent_det_fp += 1
        elif has_error and not made_change:
            sent_det_fn += 1
        elif not has_error and not made_change:
            sent_tn += 1

    # Token-level metrics
    det_precision = total_det_tp / (total_det_tp + total_det_fp) if (total_det_tp + total_det_fp) > 0 else 0
    det_recall = total_det_tp / (total_det_tp + total_det_fn) if (total_det_tp + total_det_fn) > 0 else 0
    beta = 0.5
    det_f05 = (1 + beta**2) * (det_precision * det_recall) / (beta**2 * det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

    # Sentence-level metrics
    sent_precision = sent_det_tp / (sent_det_tp + sent_det_fp) if (sent_det_tp + sent_det_fp) > 0 else 0
    sent_recall = sent_det_tp / (sent_det_tp + sent_det_fn) if (sent_det_tp + sent_det_fn) > 0 else 0
    sent_f05 = (1 + beta**2) * (sent_precision * sent_recall) / (beta**2 * sent_precision + sent_recall) if (sent_precision + sent_recall) > 0 else 0

    total_sents = len(predictions)
    sent_accuracy = (sent_tn + sent_det_tp) / total_sents if total_sents > 0 else 0

    return {
        # Token-level detection (position-based)
        'detection_token_precision': det_precision * 100,
        'detection_token_recall': det_recall * 100,
        'detection_token_f05': det_f05 * 100,

        # Sentence-level detection (binary classification)
        'detection_sent_precision': sent_precision * 100,
        'detection_sent_recall': sent_recall * 100,
        'detection_sent_f05': sent_f05 * 100,
        'detection_sent_accuracy': sent_accuracy * 100,
    }


def compute_all_metrics(predictions: List[str],
                       references: List[str],
                       sources: List[str]) -> Dict[str, float]:
    """
    Compute all available metrics.

    Metrics computed:
    GEC (Correction) Metrics:
    - correction_exact_match: Percentage of perfect predictions (0-100)
    - correction_gleu: Sentence-level BLEU with source penalty (0-100)
    - correction_bleu: Corpus-level BLEU score (0-100)
    - correction_precision: Percentage of correct corrections (0-100)
    - correction_recall: Percentage of errors fixed (0-100)
    - correction_f05: Precision-weighted F-score (0-100)
    - correction_avg_char_edit_distance: Average character-level Levenshtein distance
    - correction_median_char_edit_distance: Median character-level distance
    - correction_avg_normalized_edit_distance: Edit distance as % of reference length

    GED (Detection) Metrics:
    - detection_token_precision/recall/f05: Token-level error detection
    - detection_sent_precision/recall/f05/accuracy: Sentence-level detection

    Args:
        predictions: Model predictions
        references: Gold corrections
        sources: Original corrupted sentences

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Exact match
    metrics['correction_exact_match'] = exact_match_accuracy(predictions, references) * 100

    # N-gram overlap metrics
    metrics['correction_gleu'] = compute_gleu(predictions, references, sources)
    metrics['correction_bleu'] = compute_bleu(predictions, references)

    # GEC: F0.5 (correction metrics)
    f05_metrics = compute_f05_simple(predictions, references, sources)
    metrics.update(f05_metrics)

    # Edit distance
    edit_metrics = compute_edit_distance_metrics(predictions, references)
    metrics.update(edit_metrics)

    # GED: Error detection metrics
    ged_metrics = compute_ged_metrics(predictions, references, sources)
    metrics.update(ged_metrics)

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
