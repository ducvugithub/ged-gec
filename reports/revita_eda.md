# Revita Finnish Learner Error Data - EDA Report

**Generated:** 2026-03-14 23:14:44
**Data Source:** `data/revita/exercise_errors_Finnish.jsonl`

---

## 1. Dataset Overview

- **Total Examples:** 3,699
- **Average Snippet Length:** 20.4 words (meaningful words only)
- **Median Snippet Length:** 19 words
- **Snippet Length Range:** 10 - 99 words

---

## 2. Error Statistics

- **Total Error Targets:** 10,719
- **Total Error Instances:** 21,820
- **Average Errors per Example:** 2.90
- **Average Instances per Error:** 2.04
- **Error Range per Example:** 1 - 24

### Error Complexity (by word count)

| Type | Count | Percentage |
|------|-------|------------|
| 1 Word Error | 9,242 | 86.2% |
| 2 Word Errors | 1,204 | 11.2% |
| 3 Word Errors | 228 | 2.1% |
| 4 Word Errors | 36 | 0.3% |
| 5 Word Errors | 7 | 0.1% |
| 6 Word Errors | 2 | 0.0% |

---

## 3. Linguistic Analysis

- **Unique Meaningful Words:** 24,742
- **Total Word Occurrences:** 75,338
- **Average Word Length:** 9.90 characters
- **Unique Correct Forms:** 7,505
- **Unique Error Forms:** 15,766

### Top 20 Most Frequent Meaningful Words

| Rank | Word | Frequency |
|------|------|-----------| 
| 1 | on | 2,409 |
| 2 | ja | 2,331 |
| 3 | että | 841 |
| 4 | ei | 637 |
| 5 | myös | 527 |
| 6 | oli | 474 |
| 7 | ovat | 437 |
| 8 | se | 353 |
| 9 | voi | 327 |
| 10 | kun | 262 |
| 11 | ole | 254 |
| 12 | hän | 252 |
| 13 | mutta | 246 |
| 14 | suomen | 243 |
| 15 | paljon | 242 |
| 16 | tai | 241 |
| 17 | kuin | 240 |
| 18 | esimerkiksi | 209 |
| 19 | nyt | 203 |
| 20 | suomessa | 191 |

---

## 4. Error Count Distribution

**How many errors each example has:**

| Error Count | Examples | Percentage |
|-------------|----------|------------|
| 1 | 1,498 | 40.5% |
| 2 | 810 | 21.9% |
| 3 | 487 | 13.2% |
| 4 | 236 | 6.4% |
| 5 | 181 | 4.9% |
| 6 | 114 | 3.1% |
| 7 | 93 | 2.5% |
| 8 | 74 | 2.0% |
| 9 | 63 | 1.7% |
| 10 | 43 | 1.2% |
| 11 | 27 | 0.7% |
| 12 | 17 | 0.5% |
| 13 | 12 | 0.3% |
| 14 | 10 | 0.3% |
| 15 | 16 | 0.4% |
| 16 | 5 | 0.1% |
| 17 | 4 | 0.1% |
| 18 | 2 | 0.1% |
| 19 | 4 | 0.1% |
| 23 | 2 | 0.1% |

---

## 5. Instance Count Distribution

**How many instances each error has:**

| Instances | Error Count | Percentage |
|-----------|-------------|------------|
| 1 | 5,777 | 53.9% |
| 2 | 2,416 | 22.5% |
| 3 | 1,152 | 10.7% |
| 4 | 591 | 5.5% |
| 5 | 311 | 2.9% |
| 6 | 166 | 1.5% |
| 7 | 99 | 0.9% |
| 8 | 65 | 0.6% |
| 9 | 42 | 0.4% |
| 10 | 32 | 0.3% |
| 11 | 23 | 0.2% |
| 12 | 14 | 0.1% |
| 13 | 6 | 0.1% |
| 14 | 4 | 0.0% |
| 15 | 3 | 0.0% |

---

## 6. Error Density Distribution

**Error density = (error count / snippet length):**

| Density Range | Examples | Percentage |
|---------------|----------|------------|
| 0%-5% | 614 | 16.6% |
| 5%-10% | 1,242 | 33.6% |
| 10%-15% | 647 | 17.5% |
| 15%-20% | 320 | 8.7% |
| 20%-25% | 227 | 6.1% |
| 25%-30% | 155 | 4.2% |
| 30%-35% | 103 | 2.8% |
| 35%-40% | 98 | 2.6% |
| 40%-45% | 91 | 2.5% |
| 45%-50% | 37 | 1.0% |
| 50%-55% | 52 | 1.4% |
| 55%-60% | 32 | 0.9% |
| 60%-65% | 26 | 0.7% |
| 65%-70% | 24 | 0.6% |
| 70%-75% | 7 | 0.2% |
| 75%-80% | 9 | 0.2% |
| 80%-85% | 3 | 0.1% |
| 85%-90% | 6 | 0.2% |
| 90%-95% | 5 | 0.1% |
| 100%-105% | 1 | 0.0% |

---

## 7. Exhaustive Sampling Potential (NO LIMIT)

**If we generate ALL possible error combinations:**

- **Total Potential Samples:** 164,002,109,869,310,803,968 🤯
- **Average per Example:** 44,336,877,499,137,928
- **Median per Example:** 5
- **Range:** 2 - 163,264,770,469,377,605,632

### Examples by Potential Sample Count

| Potential Range | Examples | Percentage |
|-----------------|----------|------------|
| <100 samples | 3,033 | 82.0% |
| 100-500 samples | 201 | 5.4% |
| 500-1K samples | 40 | 1.1% |
| 1K-5K samples | 119 | 3.2% |
| 5K-10K samples | 46 | 1.2% |
| 10K-50K samples | 68 | 1.8% |
| 50K-100K samples | 26 | 0.7% |
| 100K-1M samples | 63 | 1.7% |
| >1M samples | 103 | 2.8% |

### Potential Samples by Error Count (Top 15)

| Error Count | Total Potential Samples |
|-------------|-------------------------|
| 1 | 21,820 |
| 2 | 203,904 |
| 3 | 3,229,005 |
| 4 | 61,120,662 |
| 5 | 1,120,875,218 |
| 6 | 18,102,771,654 |
| 7 | 249,840,480,141 |
| 8 | 2,927,142,914,737 |
| 9 | 29,124,882,989,222 |
| 10 | 246,576,933,009,086 |
| 11 | 1,779,063,817,634,910 |
| 12 | 10,945,742,909,819,206 |
| 13 | 57,390,856,162,752,952 |
| 14 | 255,890,204,358,843,424 |
| 15 | 966,185,102,886,392,576 |

### Snippet Length Distribution (Top 30)

| Length (words) | Examples |
|----------------|----------|
| 10 | 3 |
| 11 | 2 |
| 12 | 15 |
| 13 | 71 |
| 14 | 264 |
| 15 | 312 |
| 16 | 346 |
| 17 | 355 |
| 18 | 293 |
| 19 | 303 |
| 20 | 263 |
| 21 | 278 |
| 22 | 197 |
| 23 | 172 |
| 24 | 153 |
| 25 | 110 |
| 26 | 103 |
| 27 | 86 |
| 28 | 67 |
| 29 | 54 |
| 30 | 42 |
| 31 | 42 |
| 32 | 34 |
| 33 | 29 |
| 34 | 13 |
| 35 | 20 |
| 36 | 8 |
| 37 | 9 |
| 38 | 9 |
| 39 | 8 |

---

## 8. Error Pattern Analysis

| Error Category | Count | Percentage |
|----------------|-------|------------|
| Other | 6,180 | 28.3% |
| Spelling Errors | 5,640 | 25.8% |
| Compound Errors | 4,168 | 19.1% |
| Case Errors | 3,334 | 15.3% |
| English Interference | 2,498 | 11.4% |
| Tense Errors | 0 | 0.0% |

---

## 9. Key Insights & Recommendations


### Dataset Characteristics

- ✅ **Size**: 3,699 examples
- 📊 **Error diversity**: 21,820 error instances across 10,719 targets
- 🎯 **Avg instances per error**: 2.0

### Recommendations for Augmentation

1. **Most examples have 1 errors** - exhaustive augmentation feasible for these
2. **Most errors have 1 instance(s)** - consider in combination calculations
3. **Average error density: 14.2%** - aligns with 5-30% augmentation range
4. **Dominant error type: other** - ensure adequate representation

5. **Exhaustive strategy recommendation**: Most examples can be fully exhaustively generated