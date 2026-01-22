## exp_mis/mis-quarterly/pdf_only PDF Dataset Evaluation Report

Generated: 2026-01-22 18:48:09

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |   f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-----:|
| bib-dedupe |   11 |    0 |    1 | 1931 |                0.0057 |             0 |           0 |    0 |
| asreview   |   46 |    0 |    1 | 1896 |                0.0237 |             0 |           0 |    0 |
| buhos      |   11 |    0 |    1 | 1931 |                0.0057 |             0 |           0 |    0 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |   f1 | dataset                        | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|-----:|:-------------------------------|:-----------|
|    0 |   11 |    1 | 1931 | 0:00:04   |              0.005664 |      0.994336 |             0 |           0 |    0 | exp_mis/mis-quarterly/pdf_only | bib-dedupe |
|    0 |   46 |    1 | 1896 | 0:00:00   |              0.023687 |      0.976313 |             0 |           0 |    0 | exp_mis/mis-quarterly/pdf_only | asreview   |
|    0 |   11 |    1 | 1931 | 0:00:04   |              0.005664 |      0.994336 |             0 |           0 |    0 | exp_mis/mis-quarterly/pdf_only | buhos      |