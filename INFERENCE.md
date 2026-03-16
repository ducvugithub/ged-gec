# Model Inference and Evaluation

This guide explains how to generate predictions from trained models and evaluate them.

## Workflow

1. **Train model** → Model checkpoint saved
2. **Generate predictions** → Predictions JSONL file
3. **Evaluate predictions** → Metrics and analysis

## On Mahti (GPU)

### 1. Generate Predictions

Submit inference job:
```bash
cd /scratch/project_2006601/GEC
sbatch slurm/gec_seq2seq_inference_byt5_gpu.job
```

Monitor progress:
```bash
squeue -u $USER
tail -f logs/infer_byt5_*.log
```

Output: `predictions/byt5-small.jsonl`

### 2. Evaluate Predictions

After inference completes, run evaluation:
```bash
python -m src.evaluate \
    --predictions predictions/byt5-small.jsonl \
    --test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
    --output results/byt5-small.json
```

### 3. Download Results

Download predictions and results to local machine:
```bash
# From your local machine
rsync -avz mahti:/scratch/project_2006601/GEC/predictions/ ~/Personal/working-repos/revita/GEC/predictions/
rsync -avz mahti:/scratch/project_2006601/GEC/results/ ~/Personal/working-repos/revita/GEC/results/
```

## Locally (CPU/GPU)

If you've downloaded the model checkpoint, you can test locally:

### 1. Generate Predictions
```bash
cd ~/Personal/working-repos/revita/GEC
make infer-byt5
```

Or manually:
```bash
python -m src.models.seq2seq.inference \
    --model experiments/byt5-small \
    --test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
    --output predictions/byt5-small.jsonl \
    --batch-size 8  # Adjust based on available memory
```

### 2. Evaluate Predictions
```bash
make eval-byt5
```

Or manually:
```bash
python -m src.evaluate \
    --predictions predictions/byt5-small.jsonl \
    --test data/revita/splits/test_augmented_random_greedy_errdensity20_clean_seed42.jsonl \
    --output results/byt5-small.json
```

## Evaluation Metrics

The evaluation script computes:

- **F0.5** (precision-weighted): Primary GEC metric
- **GLEU**: GEC-specific metric based on n-gram overlap
- **Exact Match**: Percentage of perfect corrections
- **Precision & Recall**: For F0.5 breakdown
- **Edit Distance**: Character and word-level

Plus **stratified metrics** by error count:
- 0 errors
- 1 error
- 2-3 errors
- 4-5 errors
- 6+ errors

## Output Format

### Predictions File (JSONL)
```json
{
  "corrupted": "Minä menen kauppa.",
  "reference": "Minä menen kauppaan.",
  "prediction": "Minä menen kauppaan."
}
```

### Results File (JSON)
```json
{
  "aggregate": {
    "f05": 85.42,
    "gleu": 0.78,
    "exact_match": 72.3,
    "precision": 89.2,
    "recall": 79.1,
    "avg_char_edit_distance": 2.4,
    "avg_word_edit_distance": 0.8
  },
  "by_error_count": {
    "1_error": {"f05": 92.1, "count": 3245},
    "2-3_errors": {"f05": 81.5, "count": 2156}
  },
  "num_examples": 8950
}
```

## Tips

- **GPU memory**: Reduce `--batch-size` if you get OOM errors
- **Quick test**: Test on a subset first by limiting the test file
- **Beam search**: Inference uses beam_size=4 by default (slower but better quality)
- **ERRANT**: Use `--use-errant` for more accurate F0.5 (requires errant package)
