## exp_jmis/pdf_only PDF Dataset Evaluation Report

Generated: 2026-01-21 16:21:08

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |   f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-----:|
| bib-dedupe |   40 |    0 |    0 | 1603 |                0.0243 |             0 |           0 |    0 |
| asreview   |   45 |    0 |    0 | 1598 |                0.0274 |             0 |           0 |    0 |
| buhos      |   40 |    0 |    0 | 1603 |                0.0243 |             0 |           0 |    0 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |   f1 | dataset           | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|-----:|:------------------|:-----------|
|    0 |   40 |    0 | 1603 | 0:00:03   |              0.024346 |      0.975654 |             0 |           0 |    0 | exp_jmis/pdf_only | bib-dedupe |
|    0 |   45 |    0 | 1598 | 0:00:00   |              0.027389 |      0.972611 |             0 |           0 |    0 | exp_jmis/pdf_only | asreview   |
|    0 |   40 |    0 | 1603 | 0:00:04   |              0.024346 |      0.975654 |             0 |           0 |    0 | exp_jmis/pdf_only | buhos      |