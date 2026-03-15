# SLURM Job Scripts for CSC Mahti/Puhti

## Setup on Mahti

1. **Clone the repository:**
   ```bash
   git clone <repo-url> ~/GEC
   cd ~/GEC
   ```

2. **Transfer data files:**
   ```bash
   # From your local machine
   rsync -avz ~/Personal/working-repos/revita/GEC/data/ mahti:~/GEC/data/
   ```

3. **Create logs directory:**
   ```bash
   mkdir -p ~/GEC/logs
   ```

## Submit Training Jobs

### ByT5-small (300M params, recommended for Finnish)
```bash
cd ~/GEC
sbatch slurm/gec_seq2seq_train_byt5_gpu.job
```

### Monitor job status
```bash
squeue -u $USER              # Check job queue
tail -f logs/train_byt5_*.log # Follow training logs
scancel <job_id>             # Cancel a job
```

## Resources Used

- **Partition**: gpusmall (for single GPU jobs)
- **GPU**: 1x A100 (40GB)
- **Memory**: 64GB
- **Cores**: 10
- **Time limit**: 24 hours

## Expected Training Time

- ByT5-small: ~6-8 hours on A100
- mT5-base: ~4-6 hours on A100
- mBART-large: ~8-10 hours on A100

## Output

- Model checkpoints: `experiments/byt5-small/`
- Training logs: `logs/train_byt5_<job_id>.log`
