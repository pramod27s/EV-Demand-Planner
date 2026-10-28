# Results summary

## Dataset

Rows after each preprocessing step:

| step | rows |
|---|---|
| raw CSV | 259415 |
| parse start date | 259415 |
| filter 2012-2019 | 238043 |
| clean target + outliers | 236852 |
| drop leakage/identifier columns | 236852 |
| fill missing values | 236852 |
| feature engineering | 236852 |

Outlier threshold (99.5th percentile): **45.97 kWh**.

Chronological splits:

| split | rows | start | end |
|---|---|---|---|
| train | 165796 | 2012-01-01 13:24:00 | 2018-06-12 18:12:00 |
| val | 35528 | 2018-06-12 18:13:00 | 2019-04-03 12:05:00 |
| test | 35528 | 2019-04-03 12:14:00 | 2019-12-31 21:33:00 |

## Baseline models (test set, no optimisation)

| model | rmse | mae | r2 | smape | train_time_s |
|---|---|---|---|---|---|
| Linear Regression | 5.3518 | 3.5582 | 0.4073 | 45.4024 | 1.0350 |
| Random Forest | 5.4000 | 3.6245 | 0.3966 | 45.7697 | 78.5467 |
| XGBoost (default) | 5.4412 | 3.6141 | 0.3873 | 46.0942 | 2.0713 |

## Final comparison (test set)

Tuned rows use each optimiser's best seed (selected on validation RMSE); `rmse_mean`/`rmse_std` are over all seeds. `rmse_improvement_pct` is relative to default XGBoost.

| model | rmse | mae | r2 | smape | rmse_improvement_pct | train_time_s | rmse_mean | rmse_std | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| Linear Regression | 5.3518 | 3.5582 | 0.4073 | 45.4024 | 1.6436 | 1.0350 |  |  | 1 |
| Random Forest | 5.4000 | 3.6245 | 0.3966 | 45.7697 | 0.7570 | 78.5467 |  |  | 1 |
| XGBoost (default) | 5.4412 | 3.6141 | 0.3873 | 46.0942 | 0.0000 | 2.0713 |  |  | 1 |
| XGBoost + Random | 5.3054 | 3.5053 | 0.4175 | 45.1447 | 2.4954 | 11.6053 | 5.3125 | 0.0068 | 5 |
| XGBoost + GA | 5.3113 | 3.5158 | 0.4162 | 45.3667 | 2.3876 | 9.4761 | 5.3100 | 0.0031 | 5 |
| XGBoost + PSO | 5.3067 | 3.5064 | 0.4173 | 45.1656 | 2.4730 | 20.3205 | 5.3101 | 0.0049 | 5 |
| XGBoost + SMA | 5.3016 | 3.5047 | 0.4184 | 45.0786 | 2.5657 | 16.8283 | 5.3052 | 0.0038 | 5 |

## Tuning summary (across seeds)

| optimizer | n_runs | val_rmse_mean | val_rmse_std | test_rmse_mean | test_rmse_std | test_rmse_best | test_rmse_worst | test_r2_mean | runtime_s_mean | real_evaluations_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| GA | 5 | 4.8839 | 0.0076 | 5.3100 | 0.0031 | 5.3055 | 5.3137 | 0.4165 | 816.7377 | 231.4000 |
| PSO | 5 | 4.8846 | 0.0062 | 5.3101 | 0.0049 | 5.3055 | 5.3178 | 0.4165 | 1359.0314 | 300.0000 |
| Random | 5 | 4.8854 | 0.0065 | 5.3125 | 0.0068 | 5.3054 | 5.3236 | 0.4160 | 901.7004 | 300.0000 |
| SMA | 5 | 4.8879 | 0.0043 | 5.3052 | 0.0038 | 5.3016 | 5.3094 | 0.4176 | 961.8588 | 288.4000 |

## Best hyperparameters

| optimizer | seed | n_estimators | max_depth | learning_rate | subsample | colsample_bytree | min_child_weight | gamma | reg_lambda | reg_alpha | best_iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 3 | 826 | 4 | 0.0391 | 0.8068 | 0.5144 | 8.5210 | 2.0810 | 0.0096 | 0.0078 | 429 |
| GA | 3 | 754 | 3 | 0.1108 | 0.6871 | 0.5068 | 5.4500 | 4.3800 | 0.0069 | 0.3029 | 251 |
| PSO | 3 | 879 | 4 | 0.0179 | 0.7840 | 0.5000 | 8.4140 | 1.8450 | 0.0208 | 0.2104 | 878 |
| SMA | 3 | 711 | 4 | 0.0222 | 0.5000 | 0.5000 | 7.3830 | 0.0000 | 0.0010 | 0.0010 | 706 |

## Wilcoxon signed-rank test (test RMSE, paired by seed)

| comparison | metric | n_pairs | sma_mean | other_mean | sma_wins | statistic | p_value | note |
|---|---|---|---|---|---|---|---|---|
| SMA vs PSO | test_rmse | 5 | 5.3052 | 5.3101 | 4 | 1.0000 | 0.1250 | low power (few seeds) |
| SMA vs GA | test_rmse | 5 | 5.3052 | 5.3100 | 4 | 1.0000 | 0.1250 | low power (few seeds) |
| SMA vs Random | test_rmse | 5 | 5.3052 | 5.3125 | 5 | 0.0000 | 0.0625 | low power (few seeds) |

With 5 seeds the smallest possible two-sided p-value is 0.0625, so the test has low power; p-values are reported as they are.

## Benchmark functions (D = 10, 10 seeds, final best fitness)

| function | optimizer | mean | std | best | worst |
|---|---|---|---|---|---|
| Sphere | SMA | 0.01538 | 0.007954 | 0.004183 | 0.02757 |
| Sphere | PSO | 0.001923 | 0.002773 | 0.0001023 | 0.008722 |
| Sphere | GA | 0.08556 | 0.08711 | 0.003844 | 0.2192 |
| Sphere | Random | 5513 | 1586 | 1781 | 8069 |
| Rastrigin | SMA | 12 | 5.609 | 6.02 | 20.93 |
| Rastrigin | PSO | 7.861 | 2.425 | 5.144 | 12.1 |
| Rastrigin | GA | 0.4677 | 0.6943 | 0.01426 | 2.049 |
| Rastrigin | Random | 75.37 | 6.687 | 65.08 | 87.06 |
| Ackley | SMA | 0.2522 | 0.1247 | 0.09105 | 0.4401 |
| Ackley | PSO | 0.01082 | 0.003272 | 0.004835 | 0.01505 |
| Ackley | GA | 0.08525 | 0.04324 | 0.01705 | 0.1623 |
| Ackley | Random | 17.05 | 1.35 | 13.33 | 18.14 |
| Rosenbrock | SMA | 186 | 408 | 3.838 | 1325 |
| Rosenbrock | PSO | 30.55 | 35.29 | 5.898 | 84.34 |
| Rosenbrock | GA | 270.2 | 806.1 | 0.4272 | 2564 |
| Rosenbrock | Random | 3.956e+06 | 2.118e+06 | 6.023e+05 | 8.099e+06 |
| Griewank | SMA | 0.6915 | 0.1746 | 0.4697 | 0.9801 |
| Griewank | PSO | 0.2574 | 0.152 | 0.06426 | 0.4671 |
| Griewank | GA | 0.1288 | 0.04419 | 0.07459 | 0.214 |
| Griewank | Random | 50.63 | 14.28 | 17.04 | 73.62 |

## Figures

- `figures/convergence_mean.png` — Mean ± std best-so-far validation RMSE per iteration for SMA, PSO, GA and Random Search.
- `figures/convergence_sma_runs.png` — Convergence of every SMA seed.
- `figures/learning_curve_default_vs_sma.png` — Train/validation RMSE per boosting round: default vs SMA-tuned XGBoost.
- `figures/baseline_vs_optimized_bar.png` — Test RMSE, MAE and R² of the baselines and the optimised XGBoost models.
- `figures/boxplot_test_rmse.png` — Distribution of test RMSE across seeds for each optimiser.
- `figures/pred_vs_actual_sma.png` — Predicted vs actual energy on the test set (SMA-tuned XGBoost, 5,000 sessions).
- `figures/residuals_sma.png` — Residual histogram and residuals vs predicted (SMA-tuned XGBoost, test set).
- `figures/feature_importance_sma.png` — Top-15 gain importances of the SMA-tuned XGBoost.
- `figures/hyperparams_comparison.png` — Best hyperparameters of each optimiser, normalised to the search range.
- `figures/runtime_vs_rmse.png` — Mean tuning runtime vs mean test RMSE per optimiser.
- `figures/sma_weights_explained.png` — SMA weights W (vein thickness) vs fitness rank for one iteration.
- `benchmarks/*.png` — Convergence of the optimisers on Sphere, Rastrigin, Ackley, Rosenbrock and Griewank.
