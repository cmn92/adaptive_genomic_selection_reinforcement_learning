# Phenotyping Strategy Evaluation Including RL

## Evaluation design

The comparison contains 20 matched replicates, 20 breeding generations per replicate, and 5 strategies.

The strategies were evaluated using the same initial population, phenotyping budget, parent-selection rule, crossing design, and replicate seeds.

## Strategy summary

| strategy            |   replicates |   mean_total_gain |   median_total_gain |   mean_prediction_accuracy |   mean_final_variance |   mean_variance_retention |   mean_runtime_seconds |
|:--------------------|-------------:|------------------:|--------------------:|---------------------------:|----------------------:|--------------------------:|-----------------------:|
| diversity_sampling  |           20 |            6.0322 |              6.1074 |                     0.2582 |                     0 |                         0 |                 1.3516 |
| active_learning_pev |           20 |            5.7703 |              5.7666 |                     0.2304 |                     0 |                         0 |                 7.0979 |
| random_sampling     |           20 |            5.2422 |              5.2974 |                     0.2069 |                     0 |                         0 |                 1.3208 |
| fixed_sampling      |           20 |            4.9713 |              5.2216 |                     0.2054 |                     0 |                         0 |                 1.3059 |
| linear_q_learning   |           20 |            3.7682 |              3.9197 |                     0.1524 |                     0 |                         0 |                 0      |

## Main descriptive result

The highest mean total realized genetic gain was observed for **diversity_sampling**, with a mean of 6.032.

This descriptive ranking should be interpreted together with the paired statistical tests and the retained genetic variance.

## Omnibus repeated-measures tests

| metric                      | test     |   number_of_replicates |   number_of_strategies |   statistic |   p_value |   kendalls_w |
|:----------------------------|:---------|-----------------------:|-----------------------:|------------:|----------:|-------------:|
| total_realized_genetic_gain | Friedman |                     20 |                      5 |     44      |         0 |      0.55    |
| final_mean_genetic_value    | Friedman |                     20 |                      5 |     44      |         0 |      0.55    |
| mean_prediction_accuracy    | Friedman |                     20 |                      5 |     37.8    |         0 |      0.4725  |
| final_genetic_variance      | Friedman |                     20 |                      5 |     52.2353 |         0 |      0.65294 |
| variance_retention          | Friedman |                     20 |                      5 |     52.2353 |         0 |      0.65294 |
| total_cycle_seconds         | Friedman |                     20 |                      5 |     70.24   |         0 |      0.878   |

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
