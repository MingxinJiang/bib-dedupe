## exp_jit/journal-of-information-technology Evaluation Report

Generated: 2026-01-22 18:05:59

## Summary

| dataset   |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| baseline  | 1158 | 1060 |    2 | 1364 |                0.4592 |        0.9981 |      0.4779 | 0.6463 |
| pdf_only  |   13 |    0 |    0 | 1185 |                0.0109 |        0      |      0      | 0      |
| mixed     | 1375 | 1924 |   69 | 1414 |                0.493  |        0.9654 |      0.5832 | 0.7271 |

## Detailed Results

| dataset   |   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | tool       |
|:----------|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:-----------|
| baseline  | 1060 | 1158 |    2 | 1364 | 0:00:15   |              0.459159 |      0.540841 |      0.998117 |    0.477908 | 0.646341 | bib-dedupe |
| pdf_only  |    0 |   13 |    0 | 1185 | 0:00:04   |              0.010851 |      0.989149 |      0        |    0        | 0        | bib-dedupe |
| mixed     | 1924 | 1375 |   69 | 1414 | 0:00:19   |              0.493008 |      0.506992 |      0.965379 |    0.583207 | 0.727135 | bib-dedupe |