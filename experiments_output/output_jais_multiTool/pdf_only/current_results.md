## exp_jais/journal-of-the-association-for-information-systems/pdf_only PDF Dataset Evaluation Report

Generated: 2026-01-23 23:28:52

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |  138 |    7 |    6 |  777 |                0.1508 |        0.5385 |      0.0483 | 0.0886 |
| asreview   |  238 |   12 |    1 |  677 |                0.2601 |        0.9231 |      0.048  | 0.0913 |
| buhos      |   46 |    1 |   12 |  869 |                0.0503 |        0.0769 |      0.0213 | 0.0333 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                                              | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:---------------------------------------------------------------------|:-----------|
|    7 |  138 |    6 |  777 | 0:00:04   |              0.15082  |      0.84918  |      0.538462 |    0.048276 | 0.088608 | exp_jais/journal-of-the-association-for-information-systems/pdf_only | bib-dedupe |
|   12 |  238 |    1 |  677 | 0:00:00   |              0.260109 |      0.739891 |      0.923077 |    0.048    | 0.091255 | exp_jais/journal-of-the-association-for-information-systems/pdf_only | asreview   |
|    1 |   46 |   12 |  869 | 0:00:00   |              0.050273 |      0.949727 |      0.076923 |    0.021277 | 0.033333 | exp_jais/journal-of-the-association-for-information-systems/pdf_only | buhos      |