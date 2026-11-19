| model | split | AUROC | PR-AUC | Brier | ECE | NLL |
|---|---|---|---|---|---|---|
| dist_logreg | val | 0.866 | 0.870 | 0.1447 | 0.1012 | 0.4477 |
| dist_logreg | test | 0.877 | 0.886 | 0.1413 | 0.1027 | 0.4393 |
| knn | val | 0.875 | 0.910 | 0.1453 | 0.0682 | 0.4462 |
| knn | test | 0.877 | 0.916 | 0.1411 | 0.0473 | 0.4393 |
| mlp_raw | val | 0.895 | 0.924 | 0.1300 | 0.0574 | 0.4216 |
| mlp_raw | test | 0.903 | 0.932 | 0.1262 | 0.0565 | 0.4023 |
| mlp | val | 0.895 | 0.924 | 0.1261 | 0.0144 | 0.3984 |
| mlp | test | 0.903 | 0.932 | 0.1223 | 0.0215 | 0.3842 |

Within-(H,B)-cell test AUROC (discrimination beyond the controller condition; mean over cells with both labels):
- dist_logreg: mean 0.731, n-weighted 0.731
- knn: mean 0.751, n-weighted 0.751
- mlp: mean 0.853, n-weighted 0.853

Test 95% CI (cluster bootstrap over start-goal tasks):
- dist_logreg: AUROC 0.834–0.921, PR-AUC 0.823–0.938, Brier 0.1251–0.1607
- knn: AUROC 0.854–0.896, PR-AUC 0.896–0.933, Brier 0.1295–0.1545
- mlp: AUROC 0.881–0.924, PR-AUC 0.917–0.945, Brier 0.1101–0.1351
