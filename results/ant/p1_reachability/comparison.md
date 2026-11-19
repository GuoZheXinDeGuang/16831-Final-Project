| model | split | AUROC | PR-AUC | Brier | ECE | NLL |
|---|---|---|---|---|---|---|
| dist_logreg | val | 0.869 | 0.918 | 0.1441 | 0.0373 | 0.4342 |
| dist_logreg | test | 0.853 | 0.916 | 0.1464 | 0.0385 | 0.4421 |
| knn | val | 0.668 | 0.748 | 0.2188 | 0.0448 | 0.6276 |
| knn | test | 0.716 | 0.820 | 0.2022 | 0.0680 | 0.5901 |
| mlp_raw | val | 0.829 | 0.860 | 0.1661 | 0.0892 | 0.5249 |
| mlp_raw | test | 0.777 | 0.846 | 0.1957 | 0.1221 | 0.6200 |
| mlp | val | 0.829 | 0.860 | 0.1592 | 0.0482 | 0.4869 |
| mlp | test | 0.777 | 0.846 | 0.1833 | 0.0760 | 0.5483 |

Within-(H,B)-cell test AUROC (discrimination beyond the controller condition; mean over cells with both labels):
- dist_logreg: mean 0.806, n-weighted 0.806
- knn: mean 0.612, n-weighted 0.612
- mlp: mean 0.750, n-weighted 0.750

Test 95% CI (cluster bootstrap over start-goal tasks):
- dist_logreg: AUROC 0.827–0.877, PR-AUC 0.886–0.939, Brier 0.1312–0.1632
- knn: AUROC 0.695–0.736, PR-AUC 0.786–0.850, Brier 0.1948–0.2100
- mlp: AUROC 0.739–0.815, PR-AUC 0.802–0.884, Brier 0.1647–0.2038
