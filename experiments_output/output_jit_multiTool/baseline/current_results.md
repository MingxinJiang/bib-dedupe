## exp_jit/journal-of-information-technology/baseline Baseline Dataset Evaluation Report

Generated: 2026-01-23 11:27:49

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe | 1158 | 1060 |    2 | 1364 |                0.4592 |        0.9981 |      0.4779 | 0.6463 |
| asreview   |  931 | 1059 |    3 | 1591 |                0.3692 |        0.9972 |      0.5322 | 0.694  |
| buhos      |  698 |  202 |  860 | 1824 |                0.2768 |        0.1902 |      0.2244 | 0.2059 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                                            | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:---------------------------------------------------|:-----------|
| 1060 | 1158 |    2 | 1364 | 0:00:15   |              0.459159 |      0.540841 |      0.998117 |    0.477908 | 0.646341 | exp_jit/journal-of-information-technology/baseline | bib-dedupe |
| 1059 |  931 |    3 | 1591 | 0:00:00   |              0.369151 |      0.630849 |      0.997175 |    0.532161 | 0.693971 | exp_jit/journal-of-information-technology/baseline | asreview   |
|  202 |  698 |  860 | 1824 | 0:03:57   |              0.276764 |      0.723236 |      0.190207 |    0.224444 | 0.205912 | exp_jit/journal-of-information-technology/baseline | buhos      |