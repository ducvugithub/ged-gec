# Train/Val/Test Split Statistics

Combined statistics for all data splits.

======================================================================
TRAINING DATA
======================================================================

**Total samples:** 60,308

Corrupted sequences:
  Mean: 20.7 words
  Median: 19 words
  Max: 99 words
  95th percentile: 32 words
  99th percentile: 40 words

Correct sequences:
  Mean: 20.8 words
  Median: 19 words
  Max: 99 words
  95th percentile: 32 words
  99th percentile: 41 words

Estimated subword tokens (assuming 2x multiplier for Finnish):
  Max corrupted: ~198 tokens
  Max correct: ~198 tokens
  95th percentile corrupted: ~64 tokens
  99th percentile corrupted: ~80 tokens

**Error Count Distribution:**

  0 errors: 2,580 (4.3%)
  1 errors: 25,768 (42.7%)
  2 errors: 15,300 (25.4%)
  3 errors: 9,610 (15.9%)
  4 errors: 4,570 (7.6%)
  5 errors: 1,720 (2.9%)
  6 errors: 460 (0.8%)
  7 errors: 190 (0.3%)
  ... and 3 more error counts

  Mean: 1.94 errors/sample
  Median: 2 errors/sample

**Error Density Distribution:**

  0% (no errors)          2,580 (  4.3%)
  0-5%                   17,541 ( 29.1%)
  5-10%                  19,757 ( 32.8%)
  10-15%                 12,000 ( 19.9%)
  15-20%                  7,530 ( 12.5%)
  20%+                      900 (  1.5%)

  Mean: 8.14%
  Median: 6.25%

======================================================================
VALIDATION DATA
======================================================================

**Total samples:** 8,955

Corrupted sequences:
  Mean: 21.0 words
  Median: 20 words
  Max: 46 words
  95th percentile: 31 words
  99th percentile: 39 words

Correct sequences:
  Mean: 21.1 words
  Median: 20 words
  Max: 46 words
  95th percentile: 32 words
  99th percentile: 39 words

Estimated subword tokens (assuming 2x multiplier for Finnish):
  Max corrupted: ~92 tokens
  Max correct: ~92 tokens
  95th percentile corrupted: ~62 tokens
  99th percentile corrupted: ~78 tokens

**Error Count Distribution:**

  0 errors: 372 (4.2%)
  1 errors: 3,713 (41.5%)
  2 errors: 2,220 (24.8%)
  3 errors: 1,410 (15.7%)
  4 errors: 780 (8.7%)
  5 errors: 300 (3.4%)
  6 errors: 120 (1.3%)
  7 errors: 30 (0.3%)
  ... and 1 more error counts

  Mean: 2.01 errors/sample
  Median: 2 errors/sample

**Error Density Distribution:**

  0% (no errors)            372 (  4.2%)
  0-5%                    2,727 ( 30.5%)
  5-10%                   2,756 ( 30.8%)
  10-15%                  1,780 ( 19.9%)
  15-20%                  1,080 ( 12.1%)
  20%+                      240 (  2.7%)

  Mean: 8.20%
  Median: 6.45%

======================================================================
TEST DATA
======================================================================

**Total samples:** 17,644

Corrupted sequences:
  Mean: 20.5 words
  Median: 19 words
  Max: 71 words
  95th percentile: 30 words
  99th percentile: 40 words

Correct sequences:
  Mean: 20.7 words
  Median: 20 words
  Max: 71 words
  95th percentile: 30 words
  99th percentile: 41 words

Estimated subword tokens (assuming 2x multiplier for Finnish):
  Max corrupted: ~142 tokens
  Max correct: ~142 tokens
  95th percentile corrupted: ~60 tokens
  99th percentile corrupted: ~80 tokens

**Error Count Distribution:**

  0 errors: 747 (4.2%)
  1 errors: 7,457 (42.3%)
  2 errors: 4,470 (25.3%)
  3 errors: 2,840 (16.1%)
  4 errors: 1,460 (8.3%)
  5 errors: 490 (2.8%)
  6 errors: 160 (0.9%)
  7 errors: 10 (0.1%)
  ... and 1 more error counts

  Mean: 1.94 errors/sample
  Median: 2 errors/sample

**Error Density Distribution:**

  0% (no errors)            747 (  4.2%)
  0-5%                    5,093 ( 28.9%)
  5-10%                   5,704 ( 32.3%)
  10-15%                  3,570 ( 20.2%)
  15-20%                  2,290 ( 13.0%)
  20%+                      240 (  1.4%)

  Mean: 8.21%
  Median: 6.45%

======================================================================