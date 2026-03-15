# Augmented Revita GEC Data - EDA Report

**Dataset:** `train_augmented_random_greedy_errdensity20_clean_seed42.jsonl`
**Total Samples:** 60,308

---

## 1. Snippet Length Distribution

Length of snippets (in words):

| Statistic | Value |
|-----------|-------|
| Mean | 24.8 words |
| Median | 23 words |
| Std Dev | 7.7 |
| Min | 10 words |
| Max | 125 words |
| 10th percentile | 18 words |
| 25th percentile | 20 words |
| 50th percentile | 23 words |
| 75th percentile | 28 words |
| 90th percentile | 34 words |
| 95th percentile | 38 words |
| 99th percentile | 52 words |

**Length Distribution:**

| Length Range | Count | Percentage |
|--------------|-------|------------|
| 0-10 words | 21 | 0.03% |
| 10-20 words | 18,641 | 30.91% |
| 20-30 words | 32,013 | 53.08% |
| 30-40 words | 7,607 | 12.61% |
| 40-50 words | 1,374 | 2.28% |
| 50-100 words | 621 | 1.03% |
| 100+ words | 31 | 0.05% |

## 2. Error Count Distribution

Distribution of number of errors per sample:

| Errors | Count | Percentage |
|--------|-------|------------|
| 0 | 2,580 | 4.28% |
| 1 | 25,768 | 42.73% |
| 2 | 15,300 | 25.37% |
| 3 | 9,610 | 15.93% |
| 4 | 4,570 | 7.58% |
| 5 | 1,720 | 2.85% |
| 6 | 460 | 0.76% |
| 7 | 190 | 0.32% |
| 8 | 70 | 0.12% |
| 9 | 30 | 0.05% |
| 10 | 10 | 0.02% |

- **Mean errors per sample:** 1.94
- **Median:** 2
- **Range:** 0 - 10

## 3. Error Rate Distribution

Error density (errors / snippet_length):

| Error Rate | Count | Percentage |
|------------|-------|------------|
| 0% | 2,580 | 4.28% |
| 0-5% | 17,541 | 29.09% |
| 5-10% | 19,757 | 32.76% |
| 10-15% | 12,000 | 19.90% |
| 15-20% | 7,530 | 12.49% |
| 20%+ | 900 | 1.49% |

- **Mean error rate:** 8.1%
- **Median error rate:** 6.2%

## 4. Correct Frequency Distribution

Shows how many augmented samples were generated from each unique raw example.
Higher frequencies indicate that a raw sample was reused more times during augmentation.

| Statistic | Value |
|-----------|-------|
| Mean frequency | 32.0× |
| Median | 31× |
| Min | 11× |
| Max | 112× |
| Std Dev | 16.3 |

**Top 10 Most Common Frequencies:**

| Frequency | # of Texts | Total Samples |
|-----------|------------|---------------|
| 31× | 14,911 texts | 462,241 samples |
| 21× | 11,550 texts | 242,550 samples |
| 41× | 11,316 texts | 463,956 samples |
| 11× | 11,220 texts | 123,420 samples |
| 51× | 6,171 texts | 314,721 samples |
| 61× | 1,525 texts | 93,025 samples |
| 71× | 852 texts | 60,492 samples |
| 42× | 378 texts | 15,876 samples |
| 32× | 288 texts | 9,216 samples |
| 52× | 260 texts | 13,520 samples |

## 5. Edit Distance Distribution

Levenshtein distance between corrupted and correct text:

| Metric | Absolute | Normalized |
|--------|----------|------------|
| Mean | 6.91 chars | 0.040 |
| Median | 5 chars | 0.030 |
| Min | 0 chars | 0.000 |
| Max | 98 chars | 0.463 |
| Std Dev | 6.76 | 0.036 |

**Edit Distance Distribution:**

| Distance | Count | Percentage |
|----------|-------|------------|
| 0 | 2,580 | 4.28% |
| 1 | 5,304 | 8.79% |
| 2-4 | 20,259 | 33.59% |
| 5-9 | 17,939 | 29.75% |
| 10-19 | 11,274 | 18.69% |
| 20-49 | 2,820 | 4.68% |
| 50+ | 132 | 0.22% |

## 6. Edit Position Distribution

Where in the snippet do errors occur?

| Statistic | Absolute Position | Relative Position |
|-----------|-------------------|-------------------|
| Mean | 23.9 | 0.931 |
| Median | 22 | 0.944 |
| Std Dev | 16.2 | 0.533 |

**Position Distribution (relative to snippet length):**

| Position | Count | Percentage |
|----------|-------|------------|
| 0-20% | 12,707 | 10.85% |
| 20-40% | 12,414 | 10.60% |
| 40-60% | 12,369 | 10.56% |
| 60-80% | 11,631 | 9.93% |
| 80-100% | 11,527 | 9.84% |

## 7. Text Length Comparison

Corrupted vs correct text length (in characters):

| Statistic | Corrupted | Correct | Difference |
|-----------|-----------|---------|------------|
| Mean | 173.2 | 173.7 | -0.6 |
| Median | 162 | 163 | -1 |

---

## Summary

- **Total samples:** 60,308
- **Unique correct texts:** 28
- **Error count range:** 0 - 10
- **Average errors per sample:** 1.94
- **Average error rate:** 8.1%
- **Average edit distance:** 6.91 chars
- **Average snippet length:** 24.8 words
