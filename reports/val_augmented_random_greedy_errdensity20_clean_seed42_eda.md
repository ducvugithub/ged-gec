# Augmented Revita GEC Data - EDA Report

**Dataset:** `val_augmented_random_greedy_errdensity20_clean_seed42.jsonl`
**Total Samples:** 8,955

---

## 1. Snippet Length Distribution

Length of snippets (in words):

| Statistic | Value |
|-----------|-------|
| Mean | 26.2 words |
| Median | 24 words |
| Std Dev | 14.5 |
| Min | 16 words |
| Max | 171 words |
| 10th percentile | 18 words |
| 25th percentile | 20 words |
| 50th percentile | 24 words |
| 75th percentile | 28 words |
| 90th percentile | 34 words |
| 95th percentile | 38 words |
| 99th percentile | 55 words |

**Length Distribution:**

| Length Range | Count | Percentage |
|--------------|-------|------------|
| 0-10 words | 0 | 0.00% |
| 10-20 words | 2,467 | 27.55% |
| 20-30 words | 4,811 | 53.72% |
| 30-40 words | 1,323 | 14.77% |
| 40-50 words | 230 | 2.57% |
| 50-100 words | 53 | 0.59% |
| 100+ words | 71 | 0.79% |

## 2. Error Count Distribution

Distribution of number of errors per sample:

| Errors | Count | Percentage |
|--------|-------|------------|
| 0 | 372 | 4.15% |
| 1 | 3,713 | 41.46% |
| 2 | 2,220 | 24.79% |
| 3 | 1,410 | 15.75% |
| 4 | 780 | 8.71% |
| 5 | 300 | 3.35% |
| 6 | 120 | 1.34% |
| 7 | 30 | 0.34% |
| 8 | 10 | 0.11% |

- **Mean errors per sample:** 2.01
- **Median:** 2
- **Range:** 0 - 8

## 3. Error Rate Distribution

Error density (errors / snippet_length):

| Error Rate | Count | Percentage |
|------------|-------|------------|
| 0% | 372 | 4.15% |
| 0-5% | 2,727 | 30.45% |
| 5-10% | 2,756 | 30.78% |
| 10-15% | 1,780 | 19.88% |
| 15-20% | 1,080 | 12.06% |
| 20%+ | 240 | 2.68% |

- **Mean error rate:** 8.2%
- **Median error rate:** 6.5%

## 4. Correct Frequency Distribution

Shows how many augmented samples were generated from each unique raw example.
Higher frequencies indicate that a raw sample was reused more times during augmentation.

| Statistic | Value |
|-----------|-------|
| Mean frequency | 32.6× |
| Median | 31× |
| Min | 11× |
| Max | 81× |
| Std Dev | 16.0 |

**Top 10 Most Common Frequencies:**

| Frequency | # of Texts | Total Samples |
|-----------|------------|---------------|
| 41× | 1,968 texts | 80,688 samples |
| 31× | 1,860 texts | 57,660 samples |
| 21× | 1,701 texts | 35,721 samples |
| 11× | 1,639 texts | 18,029 samples |
| 51× | 867 texts | 44,217 samples |
| 61× | 549 texts | 33,489 samples |
| 71× | 142 texts | 10,082 samples |
| 81× | 81 texts | 6,561 samples |
| 50× | 50 texts | 2,500 samples |
| 42× | 42 texts | 1,764 samples |

## 5. Edit Distance Distribution

Levenshtein distance between corrupted and correct text:

| Metric | Absolute | Normalized |
|--------|----------|------------|
| Mean | 7.41 chars | 0.044 |
| Median | 5 chars | 0.032 |
| Min | 0 chars | 0.000 |
| Max | 50 chars | 0.247 |
| Std Dev | 7.04 | 0.039 |

**Edit Distance Distribution:**

| Distance | Count | Percentage |
|----------|-------|------------|
| 0 | 372 | 4.15% |
| 1 | 858 | 9.58% |
| 2-4 | 2,801 | 31.28% |
| 5-9 | 2,557 | 28.55% |
| 10-19 | 1,779 | 19.87% |
| 20-49 | 586 | 6.54% |
| 50+ | 2 | 0.02% |

## 6. Edit Position Distribution

Where in the snippet do errors occur?

| Statistic | Absolute Position | Relative Position |
|-----------|-------------------|-------------------|
| Mean | 25.5 | 0.917 |
| Median | 22 | 0.920 |
| Std Dev | 21.9 | 0.530 |

**Position Distribution (relative to snippet length):**

| Position | Count | Percentage |
|----------|-------|------------|
| 0-20% | 1,988 | 11.04% |
| 20-40% | 2,031 | 11.28% |
| 40-60% | 1,744 | 9.68% |
| 60-80% | 1,925 | 10.69% |
| 80-100% | 1,890 | 10.49% |

## 7. Text Length Comparison

Corrupted vs correct text length (in characters):

| Statistic | Corrupted | Correct | Difference |
|-----------|-----------|---------|------------|
| Mean | 174.0 | 174.6 | -0.6 |
| Median | 164 | 166 | -2 |

---

## Summary

- **Total samples:** 8,955
- **Unique correct texts:** 12
- **Error count range:** 0 - 8
- **Average errors per sample:** 2.01
- **Average error rate:** 8.2%
- **Average edit distance:** 7.41 chars
- **Average snippet length:** 26.2 words
