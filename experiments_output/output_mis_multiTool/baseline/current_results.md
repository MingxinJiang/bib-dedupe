## exp_mis/mis-quarterly/baseline Baseline Dataset Evaluation Report

Generated: 2026-01-30 11:02:31

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |    5 | 1579 |   10 | 2106 |                0.0024 |        0.9937 |      0.9968 | 0.9953 |
| asreview   |  202 |  231 | 1358 | 1909 |                0.0957 |        0.1454 |      0.5335 | 0.2285 |
| buhos      |   10 |  229 | 1360 | 2101 |                0.0047 |        0.1441 |      0.9582 | 0.2505 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                        | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:-------------------------------|:-----------|
| 1579 |    5 |   10 | 2106 | 0:00:11   |              0.002369 |      0.997631 |      0.993707 |    0.996843 | 0.995273 | exp_mis/mis-quarterly/baseline | bib-dedupe |
|  231 |  202 | 1358 | 1909 | 0:00:00   |              0.095689 |      0.904311 |      0.145374 |    0.533487 | 0.228487 | exp_mis/mis-quarterly/baseline | asreview   |
|  229 |   10 | 1360 | 2101 | 0:03:43   |              0.004737 |      0.995263 |      0.144116 |    0.958159 | 0.250547 | exp_mis/mis-quarterly/baseline | buhos      |