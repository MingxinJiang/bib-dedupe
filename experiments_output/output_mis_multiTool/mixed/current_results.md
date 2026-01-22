## exp_mis/mis-quarterly/mixed PDF Dataset Evaluation Report

Generated: 2026-01-22 18:46:04

## Tool Performance Comparison

| tool       |   FP |   TP |   FN |   TN |   false_positive_rate |   sensitivity |   precision |     f1 |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|------------:|-------:|
| bib-dedupe |   69 | 3351 |  106 | 2117 |                0.0316 |        0.9693 |      0.9798 | 0.9746 |
| asreview   |  171 | 1094 | 2363 | 2015 |                0.0782 |        0.3165 |      0.8648 | 0.4634 |
| buhos      |    0 |  242 | 3215 | 2186 |                0      |        0.07   |      1      | 0.1308 |

## Detailed Results

|   TP |   FP |   FN |   TN | runtime   |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 | dataset                     | tool       |
|-----:|-----:|-----:|-----:|:----------|----------------------:|--------------:|--------------:|------------:|---------:|:----------------------------|:-----------|
| 3351 |   69 |  106 | 2117 | 0:00:15   |              0.031565 |      0.968435 |      0.969338 |    0.979825 | 0.974553 | exp_mis/mis-quarterly/mixed | bib-dedupe |
| 1094 |  171 | 2363 | 2015 | 0:00:00   |              0.078225 |      0.921775 |      0.316459 |    0.864822 | 0.463363 | exp_mis/mis-quarterly/mixed | asreview   |
|  242 |    0 | 3215 | 2186 | 0:03:33   |              0        |      1        |      0.070003 |    1        | 0.130846 | exp_mis/mis-quarterly/mixed | buhos      |