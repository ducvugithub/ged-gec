# Augmented Revita GEC Data - EDA Report

**Dataset:** `revita_augmented_exhaustive_limit100_errdensity20_clean_seed42.jsonl`
**Total Samples:** 92,712

---

## 1. Snippet Length Distribution

Length of snippets (in words):

| Statistic | Value |
|-----------|-------|
| Mean | 24.0 words |
| Median | 23 words |
| Std Dev | 8.2 |
| Min | 10 words |
| Max | 171 words |
| 10th percentile | 18 words |
| 25th percentile | 20 words |
| 50th percentile | 23 words |
| 75th percentile | 26 words |
| 90th percentile | 32 words |
| 95th percentile | 35 words |
| 99th percentile | 48 words |

**Length Distribution:**

| Length Range | Count | Percentage |
|--------------|-------|------------|
| 0-10 words | 7 | 0.01% |
| 10-20 words | 31,160 | 33.61% |
| 20-30 words | 50,599 | 54.58% |
| 30-40 words | 8,729 | 9.42% |
| 40-50 words | 1,514 | 1.63% |
| 50-100 words | 590 | 0.64% |
| 100+ words | 113 | 0.12% |

## 2. Error Count Distribution

Distribution of number of errors per sample:

| Errors | Count | Percentage |
|--------|-------|------------|
| 0 | 3,699 | 3.99% |
| 1 | 21,593 | 23.29% |
| 2 | 48,736 | 52.57% |
| 3 | 16,036 | 17.30% |
| 4 | 2,476 | 2.67% |
| 5 | 165 | 0.18% |
| 6 | 7 | 0.01% |

- **Mean errors per sample:** 1.92
- **Median:** 2
- **Range:** 0 - 6

## 3. Error Rate Distribution

Error density (errors / snippet_length):

| Error Rate | Count | Percentage |
|------------|-------|------------|
| 0% | 3,699 | 3.99% |
| 0-5% | 15,567 | 16.79% |
| 5-10% | 39,485 | 42.59% |
| 10-15% | 26,896 | 29.01% |
| 15-20% | 6,811 | 7.35% |
| 20%+ | 254 | 0.27% |

- **Mean error rate:** 8.5%
- **Median error rate:** 8.3%

## 4. Correct Frequency Distribution

Shows how many augmented samples were generated from each unique raw example.
Higher frequencies indicate that a raw sample was reused more times during augmentation.

| Statistic | Value |
|-----------|-------|
| Mean frequency | 79.8× |
| Median | 101× |
| Min | 2× |
| Max | 202× |
| Std Dev | 36.4 |

**Top 10 Most Common Frequencies:**

| Frequency | # of Texts | Total Samples |
|-----------|------------|---------------|
| 101× | 58,984 texts | 5,957,384 samples |
| 48× | 2,160 texts | 103,680 samples |
| 2× | 1,924 texts | 3,848 samples |
| 4× | 1,792 texts | 7,168 samples |
| 8× | 1,680 texts | 13,440 samples |
| 12× | 1,656 texts | 19,872 samples |
| 72× | 1,512 texts | 108,864 samples |
| 24× | 1,488 texts | 35,712 samples |
| 6× | 1,374 texts | 8,244 samples |
| 16× | 1,328 texts | 21,248 samples |

## 5. Edit Distance Distribution

Levenshtein distance between corrupted and correct text:

| Metric | Absolute | Normalized |
|--------|----------|------------|
| Mean | 6.66 chars | 0.041 |
| Median | 5 chars | 0.034 |
| Min | 0 chars | 0.000 |
| Max | 134 chars | 0.405 |
| Std Dev | 5.44 | 0.033 |

**Edit Distance Distribution:**

| Distance | Count | Percentage |
|----------|-------|------------|
| 0 | 3,699 | 3.99% |
| 1 | 4,774 | 5.15% |
| 2-4 | 28,855 | 31.12% |
| 5-9 | 36,573 | 39.45% |
| 10-19 | 15,841 | 17.09% |
| 20-49 | 2,922 | 3.15% |
| 50+ | 48 | 0.05% |

## 6. Edit Position Distribution

Where in the snippet do errors occur?

| Statistic | Absolute Position | Relative Position |
|-----------|-------------------|-------------------|
| Mean | 21.6 | 0.901 |
| Median | 20 | 0.900 |
| Std Dev | 15.1 | 0.535 |

**Position Distribution (relative to snippet length):**

| Position | Count | Percentage |
|----------|-------|------------|
| 0-20% | 21,530 | 12.10% |
| 20-40% | 20,463 | 11.50% |
| 40-60% | 19,268 | 10.83% |
| 60-80% | 17,436 | 9.80% |
| 80-100% | 17,220 | 9.68% |

## 7. Text Length Comparison

Corrupted vs correct text length (in characters):

| Statistic | Corrupted | Correct | Difference |
|-----------|-----------|---------|------------|
| Mean | 166.3 | 167.1 | -0.8 |
| Median | 158 | 159 | -1 |

---

## Summary

- **Total samples:** 92,712
- **Unique correct texts:** 89
- **Error count range:** 0 - 6
- **Average errors per sample:** 1.92
- **Average error rate:** 8.5%
- **Average edit distance:** 6.66 chars
- **Average snippet length:** 24.0 words
