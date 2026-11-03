# Aerospace Data Acquisition Report

## Disk

- Free before acquisition: 159.03 GB
- Free after acquisition: 159.03 GB
- No N-CMAPSS archive was retained or downloaded.

## N-CMAPSS

Status: **DATA_ACCESS_UNAVAILABLE**

NASA's official PCoE repository currently reports that direct download is unavailable.
The required target remains `N-CMAPSS_DS02-006.h5` from the DS02 subset. No mirror was used.

## SMAP/MSL

Status: **AVAILABLE**

The acquired Telemanom distribution contains 82 train and
82 test channel files, with 82 label records:
{'SMAP': 55, 'MSL': 27}. The observed package contains 82 labeled channels,
not the approximate 105 anomaly-sequence figure; the observed value is preserved.
The train/test channel IDs match and all label IDs are present in the files. The label
CSV has 81 unique channel IDs because `P-2` occurs twice; this duplicate is retained
and reported rather than silently deduplicated.

Data is stored at `data/raw/SMAP_MSL/` with `train/`, `test/`, `labeled_anomalies.csv`,
and the included license. No ML training was run.

## Validation

- SMAP/MSL files: PASS
- Shape/dtype/missing-value audit: PASS
- Label-file audit: PASS
- N-CMAPSS validation: NOT RUN because the official file was unavailable
- Registry updated only for successfully validated SMAP/MSL data
