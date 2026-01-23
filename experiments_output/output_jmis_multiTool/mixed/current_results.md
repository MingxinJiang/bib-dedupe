## exp_jmis/journal-of-management-information-systems/mixed PDF Dataset Evaluation Report

Generated: 2026-01-22 18:55:36

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |  122 | 2969 |  153 | 1655 |                0.0687 |        0.951  |      0.9605 | 0.9557 |
| asreview   |  189 | 2915 |  207 | 1588 |                0.1064 |        0.9337 |      0.9391 | 0.9364 |
| buhos      |   40 |  417 | 2705 | 1737 |                0.0225 |        0.1336 |      0.9125 | 0.233  |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                                  | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:---------------------------------------------------------|:-----------|
| 2969 |  122 |  153 | 1655 | 0:00:16   |              0.068655 |      0.931345 |      0.950993 |    0.960531 | 0.955738 | exp_jmis/journal-of-management-information-systems/mixed | bib-dedupe |
| 2915 |  189 |  207 | 1588 | 0:00:00   |              0.106359 |      0.893641 |      0.933696 |    0.939111 | 0.936396 | exp_jmis/journal-of-management-information-systems/mixed | asreview   |
|  417 |   40 | 2705 | 1737 | 0:03:40   |              0.02251  |      0.97749  |      0.133568 |    0.912473 | 0.233026 | exp_jmis/journal-of-management-information-systems/mixed | buhos      |