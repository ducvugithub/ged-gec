# Augmented Revita GEC Data - EDA Report

**Dataset:** `test_augmented_random_greedy_errdensity20_clean_seed42.jsonl`
**Total Samples:** 17,644

---

## 1. Snippet Length Distribution

Length of snippets (in words):

| Statistic | Value |
|-----------|-------|
| Mean | 24.6 words |
| Median | 23 words |
| Std Dev | 7.7 |
| Min | 16 words |
| Max | 89 words |
| 10th percentile | 18 words |
| 25th percentile | 20 words |
| 50th percentile | 23 words |
| 75th percentile | 27 words |
| 90th percentile | 32 words |
| 95th percentile | 36 words |
| 99th percentile | 55 words |

**Length Distribution:**

| Length Range | Count | Percentage |
|--------------|-------|------------|
| 0-10 words | 0 | 0.00% |
| 10-20 words | 5,428 | 30.76% |
| 20-30 words | 9,739 | 55.20% |
| 30-40 words | 1,813 | 10.28% |
| 40-50 words | 438 | 2.48% |
| 50-100 words | 226 | 1.28% |
| 100+ words | 0 | 0.00% |

## 2. Error Count Distribution

Distribution of number of errors per sample:

| Errors | Count | Percentage |
|--------|-------|------------|
| 0 | 747 | 4.23% |
| 1 | 7,457 | 42.26% |
| 2 | 4,470 | 25.33% |
| 3 | 2,840 | 16.10% |
| 4 | 1,460 | 8.27% |
| 5 | 490 | 2.78% |
| 6 | 160 | 0.91% |
| 7 | 10 | 0.06% |
| 8 | 10 | 0.06% |

- **Mean errors per sample:** 1.94
- **Median:** 2
- **Range:** 0 - 8

## 3. Error Rate Distribution

Error density (errors / snippet_length):

| Error Rate | Count | Percentage |
|------------|-------|------------|
| 0% | 747 | 4.23% |
| 0-5% | 5,093 | 28.87% |
| 5-10% | 5,704 | 32.33% |
| 10-15% | 3,570 | 20.23% |
| 15-20% | 2,290 | 12.98% |
| 20%+ | 240 | 1.36% |

- **Mean error rate:** 8.2%
- **Median error rate:** 6.5%

## 4. Correct Frequency Distribution

Shows how many augmented samples were generated from each unique raw example.
Higher frequencies indicate that a raw sample was reused more times during augmentation.

| Statistic | Value |
|-----------|-------|
| Mean frequency | 31.7× |
| Median | 31× |
| Min | 11× |
| Max | 102× |
| Std Dev | 15.6 |

**Top 10 Most Common Frequencies:**

| Frequency | # of Texts | Total Samples |
|-----------|------------|---------------|
| 31× | 4,216 texts | 130,696 samples |
| 41× | 3,895 texts | 159,695 samples |
| 21× | 3,360 texts | 70,560 samples |
| 11× | 3,289 texts | 36,179 samples |
| 51× | 1,530 texts | 78,030 samples |
| 61× | 915 texts | 55,815 samples |
| 102× | 102 texts | 10,404 samples |
| 81× | 81 texts | 6,561 samples |
| 40× | 80 texts | 3,200 samples |
| 72× | 72 texts | 5,184 samples |

## 5. Edit Distance Distribution

Levenshtein distance between corrupted and correct text:

| Metric | Absolute | Normalized |
|--------|----------|------------|
| Mean | 7.18 chars | 0.042 |
| Median | 5 chars | 0.031 |
| Min | 0 chars | 0.000 |
| Max | 148 chars | 0.396 |
| Std Dev | 6.97 | 0.039 |

**Edit Distance Distribution:**

| Distance | Count | Percentage |
|----------|-------|------------|
| 0 | 747 | 4.23% |
| 1 | 1,589 | 9.01% |
| 2-4 | 5,459 | 30.94% |
| 5-9 | 5,475 | 31.03% |
| 10-19 | 3,376 | 19.13% |
| 20-49 | 978 | 5.54% |
| 50+ | 20 | 0.11% |

## 6. Edit Position Distribution

Where in the snippet do errors occur?

| Statistic | Absolute Position | Relative Position |
|-----------|-------------------|-------------------|
| Mean | 22.9 | 0.912 |
| Median | 21 | 0.900 |
| Std Dev | 15.3 | 0.530 |

**Position Distribution (relative to snippet length):**

| Position | Count | Percentage |
|----------|-------|------------|
| 0-20% | 3,516 | 10.25% |
| 20-40% | 4,045 | 11.79% |
| 40-60% | 4,047 | 11.79% |
| 60-80% | 3,584 | 10.44% |
| 80-100% | 3,609 | 10.52% |

## 7. Text Length Comparison

Corrupted vs correct text length (in characters):

| Statistic | Corrupted | Correct | Difference |
|-----------|-----------|---------|------------|
| Mean | 173.5 | 174.1 | -0.6 |
| Median | 164 | 164 | 0 |

---

## Summary

- **Total samples:** 17,644
- **Unique correct texts:** 14
- **Error count range:** 0 - 8
- **Average errors per sample:** 1.94
- **Average error rate:** 8.2%
- **Average edit distance:** 7.18 chars
- **Average snippet length:** 24.6 words
