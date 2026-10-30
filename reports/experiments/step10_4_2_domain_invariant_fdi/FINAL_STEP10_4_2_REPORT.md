# Step 10.4.2 Domain-Invariant FDI

## Protocol

Each sensor was predicted from operating-condition settings and temporal position using Ridge models fit on normal training engines only. Evidence uses condition residuals, robust residual scale, temporal persistence, and peer disagreement. Development fault scenarios selected one shared threshold; final engines and final fault scenarios were held out and used only for evaluation. Historical Step 10.x directories were not overwritten.

## Results

```
dataset                              method  threshold  precision   recall       f1      fpr      fnr  roc_auc   pr_auc  normal_windows  fault_windows
  FD001 domain_invariant_condition_residual   0.677447   0.044902 1.000000 0.085944 0.289728 0.000000 0.858541 0.045933            3524             48
  FD002 domain_invariant_condition_residual   0.677447   0.000000 0.000000 0.000000 0.010391 1.000000 0.522294 0.009066            8950             48
  FD003 domain_invariant_condition_residual   0.677447   0.013889 0.166667 0.025641 0.132154 0.833333 0.232211 0.008919            4298             48
  FD004 domain_invariant_condition_residual   0.677447   0.000000 0.000000 0.000000 0.007784 1.000000 0.764024 0.010875           10278             48
```

## Interpretation

The representation reduced FD001 false positives relative to Step 10.4.1, but the shared threshold did not provide reliable cross-dataset detection: FD002 and FD004 produced zero recall, while FD003 remained weak. Therefore this run does not demonstrate successful domain-invariant generalization. The residual representation is a measurable improvement in the failure diagnosis and should not replace the historical FD001 specialist without additional development.
