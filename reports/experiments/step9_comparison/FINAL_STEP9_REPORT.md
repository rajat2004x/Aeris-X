# Step 9: Experiment Comparison and Analysis

## Scope
This report compares completed Step 7 and Step 8 artifacts. It does not retrain models or modify prior experiment directories.

## Results

| Model | Variant | Family | MAE | RMSE | PHM | Train seconds | Inference seconds/sample | Parameters | Device |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost | reference | classical | 18.698258 | 26.407636 | 118477.10 | 11.724678 | 0.000002841 | N/A | reference |
| Random Forest | reference | classical | 20.872798 | 29.094832 | 210675.33 | 51.863329 | 0.000056136 | N/A | reference |
| Linear Regression | reference | classical | 23.115723 | 29.495626 | 166973.44 | 0.481300 | 0.000001498 | N/A | reference |
| Mean RUL | reference | classical | 48.899997 | 58.159833 | 3749212.21 | 0.000507 | 0.000000013 | N/A | reference |
| LSTM | cpu | temporal | 18.805197 | 26.913818 | 139944.81 | 1369.176877 | 0.000820229 | 206465.0 | cpu |
| GRU | cpu | temporal | 18.803086 | 26.879739 | 116200.22 | 672.472100 | 0.000387914 | 154881.0 | cpu |
| TCN | cpu | temporal | 20.893047 | 28.058901 | 190438.45 | 284.380870 | 0.000281755 | 123457.0 | cpu |
| Transformer | cpu | temporal | 17.210884 | 24.590212 | 98955.54 | 380.929213 | 0.000075803 | 68033.0 | cpu |
| Transformer | gpu | temporal | 17.793437 | 25.123281 | 61942.23 | 29.905384 | 0.000029714 | 68033.0 | NVIDIA GeForce RTX 3050 4GB Laptop GPU |

## Findings

- Best MAE and RMSE: Transformer (cpu).
- Best PHM: Transformer (gpu).
- CPU Transformer versus XGBoost MAE relative change: -7.9546%.
- Historical GPU Transformer training speedup versus CPU: 12.738x.
- Historical GPU Transformer inference speedup versus CPU: 2.551x.

## Interpretation

Transformer (cpu) has the strongest CPU accuracy and the smallest temporal parameter count; XGBoost remains much faster to train.
The historical GPU run reduced Transformer training and inference time, but produced different validation metrics and is not a like-for-like accuracy win over the CPU run.

## Limitations

- All model metrics use the same FD001 validation split, not an untouched test set.
- Timing values are historical single-run measurements and are hardware- and load-dependent.
- The GPU and CPU Transformer runs have different measured predictions, so speed comparisons do not imply identical accuracy.
- No statistical significance or repeated-seed analysis is claimed.
