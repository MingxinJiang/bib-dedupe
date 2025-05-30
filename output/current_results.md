# Summary for datasets combined

| package    |    FP |    TP |   FN |     TN |   false_positive_rate |   specificity |   sensitivity |   precision |       f1 |
|:-----------|------:|------:|-----:|-------:|----------------------:|--------------:|--------------:|------------:|---------:|
| bib-dedupe |     0 | 35036 |  229 | 125546 |             0         |      1        |      0.993506 |    1        | 0.996743 |
| buhos      |   781 |  7121 | 1000 |  19048 |             0.0393868 |      0.960613 |      0.876862 |    0.901164 | 0.888847 |
| asreview   |  5522 | 29755 | 5510 | 120024 |             0.0439839 |      0.956016 |      0.843754 |    0.843467 | 0.843611 |
| asysd      | 55834 | 25105 |    0 |      0 |             1         |      0        |      1        |    0.310172 | 0.473483 |

# Individual datasets

## respiratory (n=1988)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 |  408 |   28 | 1552 |                0      |        1      |        0.9358 |      1      | 0.9668 | 0:00:09   |
| asreview   |   14 |  399 |   37 | 1538 |                0.009  |        0.991  |        0.9151 |      0.9661 | 0.9399 | 0:00:00   |
| buhos      |   13 |  395 |   41 | 1539 |                0.0084 |        0.9916 |        0.906  |      0.9681 | 0.936  | 0:05:09   |
| asysd      | 1552 |  436 |    0 |    0 |                1      |        0      |        1      |      0.2193 | 0.3597 | 0:00:02   |

## digital_work (n=7159)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 |  368 |    1 | 6790 |                0      |        1      |        0.9973 |      1      | 0.9986 | 0:00:36   |
| buhos      |   66 |  306 |   63 | 6724 |                0.0097 |        0.9903 |        0.8293 |      0.8226 | 0.8259 | 0:29:53   |
| asreview   |  217 |  336 |   33 | 6573 |                0.032  |        0.968  |        0.9106 |      0.6076 | 0.7289 | 0:00:00   |
| asysd      | 6790 |  369 |    0 |    0 |                1      |        0      |        1      |      0.0515 | 0.098  | 0:00:02   |

## haematology (n=1415)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 |  120 |   15 | 1278 |                0      |        1      |        0.8889 |      1      | 0.9412 | 0:00:06   |
| buhos      |   18 |  108 |   27 | 1260 |                0.0141 |        0.9859 |        0.8    |      0.8571 | 0.8276 | 0:03:32   |
| asreview   |   43 |  106 |   29 | 1235 |                0.0336 |        0.9664 |        0.7852 |      0.7114 | 0.7465 | 0:00:00   |
| asysd      | 1278 |  135 |    0 |    0 |                1      |        0      |        1      |      0.0955 | 0.1744 | 0:00:02   |

## stroke (n=1292)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 |  312 |    2 |  978 |                0      |        1      |        0.9936 |      1      | 0.9968 | 0:00:05   |
| asreview   |   12 |  309 |    5 |  966 |                0.0123 |        0.9877 |        0.9841 |      0.9626 | 0.9732 | 0:00:00   |
| buhos      |    8 |  303 |   11 |  970 |                0.0082 |        0.9918 |        0.965  |      0.9743 | 0.9696 | 0:05:14   |
| asysd      |  978 |  314 |    0 |    0 |                1      |        0      |        1      |      0.243  | 0.391  | 0:00:02   |

## cytology_screening (n=1856)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 |  766 |    6 | 1084 |                0      |        1      |        0.9922 |      1      | 0.9961 | 0:00:07   |
| asreview   |   12 |  753 |   19 | 1072 |                0.0111 |        0.9889 |        0.9754 |      0.9843 | 0.9798 | 0:00:00   |
| buhos      |   29 |  741 |   31 | 1055 |                0.0268 |        0.9732 |        0.9598 |      0.9623 | 0.9611 | 0:06:47   |
| asysd      | 1084 |  772 |    0 |    0 |                1      |        0      |        1      |      0.4159 | 0.5875 | 0:00:02   |

## srsr (n=53001)

| package    |    FP |    TP |   FN |    TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|------:|------:|-----:|------:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |     0 | 16916 |   68 | 36005 |                0      |        1      |        0.996  |      1      | 0.998  | 0:07:39   |
| asreview   |  1769 | 16061 |  923 | 34236 |                0.0491 |        0.9509 |        0.9457 |      0.9008 | 0.9227 | 0:00:02   |
| asysd      | 36005 | 16984 |    0 |     0 |                1      |        0      |        1      |      0.3205 | 0.4854 | 0:00:07   |

## diabetes (n=1845)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 | 1261 |    0 |  584 |                0      |        1      |        1      |      1      | 1      | 0:00:08   |
| asreview   |   24 | 1251 |   10 |  560 |                0.0411 |        0.9589 |        0.9921 |      0.9812 | 0.9866 | 0:00:00   |
| buhos      |  121 | 1141 |  120 |  463 |                0.2072 |        0.7928 |        0.9048 |      0.9041 | 0.9045 | 0:08:16   |
| asysd      |  584 | 1261 |    0 |    0 |                1      |        0      |        1      |      0.6835 | 0.812  | 0:00:02   |

## depression (n=79880)

| package    |   FP |    TP |   FN |    TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|------:|-----:|------:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 | 10072 |   88 | 69712 |                0      |        1      |        0.9913 |      1      | 0.9957 | 0:09:45   |
| asreview   | 3297 |  6114 | 4046 | 66415 |                0.0473 |        0.9527 |        0.6018 |      0.6497 | 0.6248 | 0:00:03   |

## neuroimaging (n=3438)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 | 1292 |    7 | 2139 |                0      |        1      |        0.9946 |      1      | 0.9973 | 0:00:16   |
| asreview   |    8 | 1261 |   38 | 2131 |                0.0037 |        0.9963 |        0.9707 |      0.9937 | 0.9821 | 0:00:00   |
| buhos      |  211 | 1065 |  234 | 1928 |                0.0986 |        0.9014 |        0.8199 |      0.8346 | 0.8272 | 0:15:35   |
| asysd      | 2139 | 1299 |    0 |    0 |                1      |        0      |        1      |      0.3778 | 0.5484 | 0:00:02   |

## cardiac (n=8948)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| bib-dedupe |    0 | 3520 |   12 | 5415 |                0      |        1      |        0.9966 |      1      | 0.9983 | 0:00:52   |
| asreview   |  124 | 3163 |  369 | 5291 |                0.0229 |        0.9771 |        0.8955 |      0.9623 | 0.9277 | 0:00:00   |
| buhos      |  313 | 3061 |  471 | 5102 |                0.0578 |        0.9422 |        0.8666 |      0.9072 | 0.8865 | 1:13:45   |
| asysd      | 5415 | 3532 |    0 |    0 |                1      |        0      |        1      |      0.3948 | 0.5661 | 0:00:03   |

## special_cases (n=6)

| package    |   FP |   TP |   FN |   TN |   false_positive_rate |   specificity |   sensitivity |   precision |     f1 | runtime   |
|:-----------|-----:|-----:|-----:|-----:|----------------------:|--------------:|--------------:|------------:|-------:|:----------|
| asreview   |    2 |    2 |    1 |    7 |                0.2222 |        0.7778 |        0.6667 |      0.5    | 0.5714 | 0:00:00   |
| bib-dedupe |    0 |    1 |    2 |    9 |                0      |        1      |        0.3333 |      1      | 0.5    | 0:00:00   |
| asysd      |    9 |    3 |    0 |    0 |                1      |        0      |        1      |      0.25   | 0.4    | 0:00:02   |
| buhos      |    2 |    1 |    2 |    7 |                0.2222 |        0.7778 |        0.3333 |      0.3333 | 0.3333 | 0:00:00   |

