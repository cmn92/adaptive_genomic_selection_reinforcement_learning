# Phenotyping Strategy Evaluation Including RL

## Evaluation design

The comparison contains 20 matched replicates, 8 breeding generations per replicate, and 5 strategies.

The strategies were evaluated using the same initial population, phenotyping budget, parent-selection rule, crossing design, and replicate seeds.

## Strategy summary

| strategy            |   replicates |   mean_total_gain |   median_total_gain |   mean_prediction_accuracy |   mean_final_variance |   mean_variance_retention |   mean_runtime_seconds |
|:--------------------|-------------:|------------------:|--------------------:|---------------------------:|----------------------:|--------------------------:|-----------------------:|
| active_learning_pev |           20 |            2.045  |              2.0831 |                     0.1547 |                0.1221 |                    0.1315 |                 2.2649 |
| linear_q_learning   |           20 |            1.9569 |              2.0962 |                     0.1898 |                0.096  |                    0.1022 |                 0      |
| diversity_sampling  |           20 |            1.9562 |              2.0382 |                     0.1709 |                0.111  |                    0.1172 |                 0.4481 |
| fixed_sampling      |           20 |            1.8153 |              1.6623 |                     0.1489 |                0.0615 |                    0.0645 |                 0.4588 |
| random_sampling     |           20 |            1.549  |              1.6294 |                     0.1341 |                0.0905 |                    0.0982 |                 0.4589 |

## Main descriptive result

The highest mean total realized genetic gain was observed for **active_learning_pev**, with a mean of 2.045.

This descriptive ranking should be interpreted together with the paired statistical tests and the retained genetic variance.

## Omnibus repeated-measures tests

| metric                      | test     |   number_of_replicates |   number_of_strategies |   statistic |   p_value |   kendalls_w |
|:----------------------------|:---------|-----------------------:|-----------------------:|------------:|----------:|-------------:|
| total_realized_genetic_gain | Friedman |                     20 |                      5 |        6.96 |   0.13802 |       0.087  |
| final_mean_genetic_value    | Friedman |                     20 |                      5 |        6.96 |   0.13802 |       0.087  |
| mean_prediction_accuracy    | Friedman |                     20 |                      5 |        8.52 |   0.07428 |       0.1065 |
| final_genetic_variance      | Friedman |                     20 |                      5 |        7.52 |   0.11083 |       0.094  |
| variance_retention          | Friedman |                     20 |                      5 |        7.52 |   0.11083 |       0.094  |
| total_cycle_seconds         | Friedman |                     20 |                      5 |       64.64 |   0       |       0.808  |

## Figures

### Mean Genetic Value

![Mean Genetic Value](figures/mean_genetic_value.png)

### Prediction Accuracy

![Prediction Accuracy](figures/prediction_accuracy.png)

### Genetic Variance

![Genetic Variance](figures/genetic_variance.png)

### Total Genetic Gain Boxplot

![Total Genetic Gain Boxplot](figures/total_genetic_gain_boxplot.png)

### Variance Retention Boxplot

![Variance Retention Boxplot](figures/variance_retention_boxplot.png)

### Runtime Boxplot

![Runtime Boxplot](figures/runtime_boxplot.png)

### Gain Variance Tradeoff

![Gain Variance Tradeoff](figures/gain_variance_tradeoff.png)

## Output tables

- `raw/generation_results.csv`: one row per strategy, replicate, and generation.
- `raw/replicate_results.csv`: one row per strategy and replicate.
- `processed/descriptive_statistics.csv`: summary statistics for each outcome.
- `processed/friedman_tests.csv`: overall repeated-measures tests.
- `processed/pairwise_tests.csv`: paired Wilcoxon tests with Holm correction and effect sizes.

## Interpretation cautions

The development comparison with only a few replicates is a pipeline check, not the final scientific analysis. Final conclusions should use a larger number of matched replicates. A strategy should not be judged on genetic gain alone because rapid gain may be accompanied by faster depletion of genetic variance.
