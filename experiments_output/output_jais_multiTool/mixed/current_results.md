## exp_jais/journal-of-the-association-for-information-systems/mixed PDF Dataset Evaluation Report

Generated: 2026-01-23 23:33:51

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |    5 | 1539 |  251 |  943 |                0.0053 |        0.8598 |      0.9968 | 0.9232 |
| asreview   |    8 | 1417 |  373 |  940 |                0.0084 |        0.7916 |      0.9944 | 0.8815 |
| buhos      |    1 |  130 | 1660 |  947 |                0.0011 |        0.0726 |      0.9924 | 0.1353 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                                           | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:------------------------------------------------------------------|:-----------|
| 1539 |    5 |  251 |  943 | 0:00:08   |              0.005274 |      0.994726 |      0.859777 |    0.996762 | 0.923215 | exp_jais/journal-of-the-association-for-information-systems/mixed | bib-dedupe |
| 1417 |    8 |  373 |  940 | 0:00:00   |              0.008439 |      0.991561 |      0.79162  |    0.994386 | 0.881493 | exp_jais/journal-of-the-association-for-information-systems/mixed | asreview   |
|  130 |    1 | 1660 |  947 | 0:02:49   |              0.001055 |      0.998945 |      0.072626 |    0.992366 | 0.135346 | exp_jais/journal-of-the-association-for-information-systems/mixed | buhos      |