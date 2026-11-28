| model | split | AUROC | PR-AUC | Brier | ECE | NLL |
|---|---|---|---|---|---|---|
| dist_logreg | val | 0.884 | 0.926 | 0.1321 | 0.0212 | 0.4084 |
| dist_logreg | test | 0.884 | 0.926 | 0.1321 | 0.0212 | 0.4084 |
| knn | val | 0.685 | 0.773 | 0.2104 | 0.0184 | 0.6087 |
| knn | test | 0.685 | 0.773 | 0.2104 | 0.0184 | 0.6087 |
| mlp_raw | val | 0.794 | 0.851 | 0.1806 | 0.0723 | 0.5476 |
| mlp_raw | test | 0.794 | 0.851 | 0.1806 | 0.0723 | 0.5476 |
| mlp | val | 0.794 | 0.851 | 0.1772 | 0.0717 | 0.5313 |
| mlp | test | 0.794 | 0.851 | 0.1772 | 0.0717 | 0.5313 |

Within-(H,B)-cell test AUROC (discrimination beyond the controller condition; mean over cells with both labels):
- dist_logreg: mean 0.862, n-weighted 0.862
- knn: mean 0.568, n-weighted 0.568
- mlp: mean 0.793, n-weighted 0.793

Test 95% CI (cluster bootstrap over start-goal tasks):
- dist_logreg: AUROC 0.860–0.907, PR-AUC 0.884–0.956, Brier 0.1122–0.1536
- knn: AUROC 0.638–0.733, PR-AUC 0.666–0.853, Brier 0.1879–0.2372
- mlp: AUROC 0.722–0.853, PR-AUC 0.793–0.900, Brier 0.1517–0.2014
