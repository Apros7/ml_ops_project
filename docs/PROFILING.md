# Profiling Guide

This project includes profiling utilities to help optimize performance. Profiling can be enabled for training loops and data processing.

## Enabling Profiling

Profiling is controlled by the `ENABLE_PROFILING` environment variable and config options:

```bash
# Enable profiling via environment variable
export ENABLE_PROFILING=true

# Or set it when running commands
ENABLE_PROFILING=true uv run python -m ml_ops.train train-ocr data/
```

## Configuration

Profiling can also be enabled via Hydra config:

```yaml
# configs/training/ocr/default.yaml
enable_profiling: true
profile_every_n_steps: 100  # Profile every 100 training steps
```

## What Gets Profiled

### 1. Training Steps (`model.py`)

- **PyTorch Profiler** is used to profile training and validation steps
- Profiles forward pass, loss computation, and decoding
- Only profiles every N steps (configurable) to avoid overhead
- Outputs Chrome trace files (`.json`) that can be viewed in Chrome's `chrome://tracing`

**Location**: `src/ml_ops/model.py` - `PlateOCR.training_step()` and `validation_step()`

### 2. Data Processing (`data.py`)

- **cProfile** is used to profile data export functions
- Profiles image loading, parsing, and file I/O operations
- Outputs stats files (`.stats`) with function-level timing

**Location**: `src/ml_ops/data.py` - `export_yolo_format()`

## Output Files

Profiling results are saved to:
- Default: `runs/profiling/`
- Can be customized via `PROFILE_OUTPUT_DIR` environment variable

### PyTorch Profiler Output
- `train_step_{batch_idx}.json` - Chrome trace files
- View in Chrome: Navigate to `chrome://tracing` and load the JSON file

### cProfile Output
- `export_yolo_format.stats` - Text file with function statistics
- `{profile_name}.stats` - Other profile outputs

## Usage Examples

### Profile OCR Training

```bash
# Enable profiling and train
ENABLE_PROFILING=true uv run python -m ml_ops.train train-ocr data/ccpd_small --max-images 1000

# Or with config override
ENABLE_PROFILING=true uv run python -m ml_ops.train train-ocr data/ccpd_small -o training/ocr.enable_profiling=true
```

### Profile Data Export

```bash
# Profile data export
ENABLE_PROFILING=true uv run python -m ml_ops.data preprocess data/raw data/processed
```

### View PyTorch Profiler Results

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select the `.json` trace file from `runs/profiling/`
4. Explore the timeline to identify bottlenecks

### View cProfile Results

```bash
# View stats file
cat runs/profiling/export_yolo_format.stats

# Or use Python to analyze
python -m pstats runs/profiling/export_yolo_format.stats
```

## Performance Impact

- Profiling adds overhead (typically 5-20% slower)
- Only profile when needed for optimization
- Use `profile_every_n_steps` to reduce overhead (default: every 100 steps)
- Validation profiling only runs on first batch of each epoch

## Best Practices

1. **Profile selectively**: Only enable profiling when investigating performance issues
2. **Profile representative workloads**: Use typical batch sizes and data sizes
3. **Compare before/after**: Profile before and after optimizations to measure improvement
4. **Focus on hotspots**: Look for functions that take the most time
5. **Check data loading**: Profile data processing separately from training

## Profiling Tools

The project uses:
- **PyTorch Profiler**: For PyTorch operations (training, inference)
- **cProfile**: For general Python code (data processing, file I/O)

Both are built-in Python/PyTorch tools - no additional dependencies required.
