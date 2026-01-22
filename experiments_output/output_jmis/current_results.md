## exp_jmis/journal-of-management-information-systems Evaluation Report

Generated: 2026-01-22 18:09:09

## Summary

| dataset   |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| baseline  |   73 | 1532 |   20 | 1653 |                0.0423 |        0.9871 |      0.9545 | 0.9705 |
| pdf_only  |   18 |    0 |    0 | 1603 |                0.0111 |        0      |      0      | 0      |
| mixed     |  122 | 2969 |  153 | 1655 |                0.0687 |        0.951  |      0.9605 | 0.9557 |

## Detailed Results

| dataset   |   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | tool       |
|:----------|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:-----------|
| baseline  | 1532 |   73 |   20 | 1653 | 0:00:14   |              0.042294 |      0.957706 |      0.987113 |    0.954517 | 0.970542 | bib-dedupe |
| pdf_only  |    0 |   18 |    0 | 1603 | 0:00:04   |              0.011104 |      0.988896 |      0        |    0        | 0        | bib-dedupe |
| mixed     | 2969 |  122 |  153 | 1655 | 0:00:17   |              0.068655 |      0.931345 |      0.950993 |    0.960531 | 0.955738 | bib-dedupe |