## exp_jmis/journal-of-management-information-systems/baseline Baseline Dataset Evaluation Report

Generated: 2026-01-22 19:00:35

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |   73 | 1532 |   20 | 1653 |                0.0423 |        0.9871 |      0.9545 | 0.9705 |
| asreview   |  203 | 1531 |   21 | 1523 |                0.1176 |        0.9865 |      0.8829 | 0.9318 |
| buhos      |   65 |  388 | 1164 | 1661 |                0.0377 |        0.25   |      0.8565 | 0.387  |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                                     | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:------------------------------------------------------------|:-----------|
| 1532 |   73 |   20 | 1653 | 0:00:12   |              0.042294 |      0.957706 |      0.987113 |    0.954517 | 0.970542 | exp_jmis/journal-of-management-information-systems/baseline | bib-dedupe |
| 1531 |  203 |   21 | 1523 | 0:00:00   |              0.117613 |      0.882387 |      0.986469 |    0.88293  | 0.931832 | exp_jmis/journal-of-management-information-systems/baseline | asreview   |
|  388 |   65 | 1164 | 1661 | 0:03:41   |              0.037659 |      0.962341 |      0.25     |    0.856512 | 0.387032 | exp_jmis/journal-of-management-information-systems/baseline | buhos      |