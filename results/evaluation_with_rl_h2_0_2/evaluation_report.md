# Phenotyping Strategy Evaluation Including RL

## Evaluation design

The comparison contains 20 matched replicates, 8 breeding generations per replicate, and 5 strategies.

The strategies were evaluated using the same initial population, phenotyping budget, parent-selection rule, crossing design, and replicate seeds.

## Strategy summary

| strategy            |   replicates |   mean_total_gain |   median_total_gain |   mean_prediction_accuracy |   mean_final_variance |   mean_variance_retention |   mean_runtime_seconds |
|:--------------------|-------------:|------------------:|--------------------:|---------------------------:|----------------------:|--------------------------:|-----------------------:|
| active_learning_pev |           20 |            3.6754 |              3.789  |                     0.3203 |                0.0782 |                    0.0826 |                 2.2109 |
| diversity_sampling  |           20 |            3.537  |              3.58   |                     0.3101 |                0.0909 |                    0.0964 |                 0.4505 |
| linear_q_learning   |           20 |            3.1277 |              3.0249 |                     0.283  |                0.0591 |                    0.0623 |                 0      |
| fixed_sampling      |           20 |            2.9551 |              2.893  |                     0.2879 |                0.0538 |                    0.0574 |                 0.4477 |
| random_sampling     |           20 |            2.6191 |              2.6075 |                     0.2657 |                0.0408 |                    0.0436 |                 0.4602 |

## Main descriptive result

The highest mean total realized genetic gain was observed for **active_learning_pev**, with a mean of 3.675.

This descriptive ranking should be interpreted together with the paired statistical tests and the retained genetic variance.

## Omnibus repeated-measures tests

| metric                      | test     |   number_of_replicates |   number_of_strategies |   statistic |   p_value |   kendalls_w |
|:----------------------------|:---------|-----------------------:|-----------------------:|------------:|----------:|-------------:|
| total_realized_genetic_gain | Friedman |                     20 |                      5 |       17.24 |   0.00174 |       0.2155 |
| final_mean_genetic_value    | Friedman |                     20 |                      5 |       17.24 |   0.00174 |       0.2155 |
| mean_prediction_accuracy    | Friedman |                     20 |                      5 |        8.8  |   0.0663  |       0.11   |
| final_genetic_variance      | Friedman |                     20 |                      5 |       11.6  |   0.02059 |       0.145  |
| variance_retention          | Friedman |                     20 |                      5 |       11.6  |   0.02059 |       0.145  |
| total_cycle_seconds         | Friedman |                     20 |                      5 |       64.12 |   0       |       0.8015 |

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
