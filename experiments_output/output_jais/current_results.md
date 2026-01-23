## exp_jais/journal-of-the-association-for-information-systems Evaluation Report

Generated: 2026-01-23 17:41:28

## Summary

| dataset   |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| baseline  |    1 |  870 |    3 |  936 |                0.0011 |        0.9966 |      0.9989 | 0.9977 |
| pdf_only  |  138 |    7 |    6 |  777 |                0.1508 |        0.5385 |      0.0483 | 0.0886 |
| mixed     |    5 | 1539 |  251 |  943 |                0.0053 |        0.8598 |      0.9968 | 0.9232 |

## Detailed Results

| dataset   |   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | tool       |
|:----------|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:-----------|
| baseline  |  870 |    1 |    3 |  936 | 0:00:14   |              0.001067 |      0.998933 |      0.996564 |    0.998852 | 0.997706 | bib-dedupe |
| pdf_only  |    7 |  138 |    6 |  777 | 0:00:10   |              0.15082  |      0.84918  |      0.538462 |    0.048276 | 0.088608 | bib-dedupe |
| mixed     | 1539 |    5 |  251 |  943 | 0:00:18   |              0.005274 |      0.994726 |      0.859777 |    0.996762 | 0.923215 | bib-dedupe |