## exp_jit/journal-of-information-technology/pdf_only PDF Dataset Evaluation Report

Generated: 2026-01-23 11:28:12

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |   f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-----:|
| bib-dedupe |   13 |    0 |    0 | 1185 |                0.0109 |             0 |           0 |    0 |
| asreview   |   45 |    0 |    0 | 1153 |                0.0376 |             0 |           0 |    0 |
| buhos      |    1 |    0 |    0 | 1197 |                0.0008 |             0 |           0 |    0 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |   f1 | dataset                                            | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|-----:|:---------------------------------------------------|:-----------|
|    0 |   13 |    0 | 1185 | 0:00:03   |              0.010851 |      0.989149 |             0 |           0 |    0 | exp_jit/journal-of-information-technology/pdf_only | bib-dedupe |
|    0 |   45 |    0 | 1153 | 0:00:00   |              0.037563 |      0.962437 |             0 |           0 |    0 | exp_jit/journal-of-information-technology/pdf_only | asreview   |
|    0 |    1 |    0 | 1197 | 0:00:02   |              0.000835 |      0.999165 |             0 |           0 |    0 | exp_jit/journal-of-information-technology/pdf_only | buhos      |