## exp_jais/journal-of-the-association-for-information-systems/baseline Baseline Dataset Evaluation Report

Generated: 2026-01-23 22:45:31

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |    1 |  870 |    3 |  936 |                0.0011 |        0.9966 |      0.9989 | 0.9977 |
| asreview   |    0 |  673 |  200 |  937 |                0      |        0.7709 |      1      | 0.8706 |
| buhos      |    1 |   79 |  794 |  936 |                0.0011 |        0.0905 |      0.9875 | 0.1658 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                                              | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:---------------------------------------------------------------------|:-----------|
|  870 |    1 |    3 |  936 | 0:00:06   |              0.001067 |      0.998933 |      0.996564 |    0.998852 | 0.997706 | exp_jais/journal-of-the-association-for-information-systems/baseline | bib-dedupe |
|  673 |    0 |  200 |  937 | 0:00:00   |              0        |      1        |      0.770905 |    1        | 0.870634 | exp_jais/journal-of-the-association-for-information-systems/baseline | asreview   |
|   79 |    1 |  794 |  936 | 0:03:17   |              0.001067 |      0.998933 |      0.090493 |    0.9875   | 0.165792 | exp_jais/journal-of-the-association-for-information-systems/baseline | buhos      |