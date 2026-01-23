## exp_jit/journal-of-information-technology/mixed PDF Dataset Evaluation Report

Generated: 2026-01-23 11:37:12

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe | 1375 | 1924 |   69 | 1414 |                0.493  |        0.9654 |      0.5832 | 0.7271 |
| asreview   | 1105 | 1946 |   47 | 1684 |                0.3962 |        0.9764 |      0.6378 | 0.7716 |
| buhos      |  731 |  361 | 1632 | 2058 |                0.2621 |        0.1811 |      0.3306 | 0.234  |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                         | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:------------------------------------------------|:-----------|
| 1924 | 1375 |   69 | 1414 | 0:00:17   |              0.493008 |      0.506992 |      0.965379 |    0.583207 | 0.727135 | exp_jit/journal-of-information-technology/mixed | bib-dedupe |
| 1946 | 1105 |   47 | 1684 | 0:00:00   |              0.396199 |      0.603801 |      0.976417 |    0.637824 | 0.77161  | exp_jit/journal-of-information-technology/mixed | asreview   |
|  361 |  731 | 1632 | 2058 | 0:04:38   |              0.262101 |      0.737899 |      0.181134 |    0.330586 | 0.234036 | exp_jit/journal-of-information-technology/mixed | buhos      |