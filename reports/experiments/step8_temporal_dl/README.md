# Step 8: Temporal Deep Learning Benchmark

## Objective
Measure whether sequence models improve RUL prediction over the Step 7 classical reference.

## Dataset and input representation
The leakage-safe FD001 arrays from Step 6 were reused without refitting scalers or regenerating windows.
Temporal models received `(batch, 30, 15)` windows. Classical models use flattened windows.

## Models and methodology
LSTM, GRU, residual dilated TCN, and compact Transformer Encoder models were trained sequentially
with Adam, MSELoss, gradient clipping, early stopping, and validation-loss checkpointing.
The validation set was used only for evaluation and validation-based checkpoint selection.

## Results

```text
      model             model_type       mae      rmse     phm_score  training_time_seconds  inference_time_seconds  inference_time_per_sample  parameter_count  trainable_parameter_count  model_size_mb    device
    XGBoost    classical_reference 18.698258 26.407636 118477.096235              11.724678                0.007463                   0.000003              NaN                        NaN       1.316036 reference
       LSTM temporal_deep_learning 18.805197 26.913818 139944.810847            1369.176877                2.154741                   0.000820         206465.0                   206465.0       0.787601       cpu
        GRU temporal_deep_learning 18.803086 26.879739 116200.217321             672.472100                1.019051                   0.000388         154881.0                   154881.0       0.590824       cpu
        TCN temporal_deep_learning 20.893047 28.058901 190438.446282             284.380870                0.740171                   0.000282         123457.0                   123457.0       0.470951       cpu
Transformer temporal_deep_learning 17.210884 24.590212  98955.540074             380.929213                0.199135                   0.000076          68033.0                    68033.0       0.259525       cpu
```

## Comparison with XGBoost
The Step 7 XGBoost reference has MAE 18.6983.
The best temporal model by MAE was Transformer with MAE 17.2109.

## Limitations and conclusion
This is a validation benchmark, not an untouched test estimate. No statistical significance claim is made.
The conclusion should be revisited with an untouched test split and repeated seeds.
