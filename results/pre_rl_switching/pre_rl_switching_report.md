# Pre-RL Switching Analysis

## Phase 1: Is Switching Useful?

```text
                     scenario  number_of_generations  number_of_unique_winners  number_of_strategy_switches  switching_detected                                                                                                                                                         winner_sequence
h2_0.20_budget_200_parents_20                      8                         3                            5                True diversity_sampling -> fixed_sampling -> diversity_sampling -> fixed_sampling -> diversity_sampling -> active_learning_pev -> active_learning_pev -> active_learning_pev
h2_0.40_budget_200_parents_20                      8                         4                            5                True          active_learning_pev -> random_sampling -> highest_gebv -> diversity_sampling -> highest_gebv -> diversity_sampling -> diversity_sampling -> diversity_sampling
h2_0.70_budget_200_parents_20                      8                         4                            4                True            highest_gebv -> diversity_sampling -> diversity_sampling -> fixed_sampling -> random_sampling -> random_sampling -> diversity_sampling -> diversity_sampling
```

## Phase 2: Best Fixed Strategy vs Oracle Switching

```text
                     scenario best_fixed_strategy  best_fixed_mean_total_gain  oracle_mean_total_gain  oracle_gain_advantage
h2_0.20_budget_200_parents_20 active_learning_pev                    4.859331                5.360128               0.500796
h2_0.40_budget_200_parents_20  diversity_sampling                    5.778395                5.800185               0.021790
h2_0.70_budget_200_parents_20  diversity_sampling                    6.791534                6.841541               0.050006
```

## Interpretation

If the winner sequence rarely changes, a fixed heuristic may be enough. If oracle switching has little advantage over the best fixed strategy, RL has little room to improve. If winner sequences change and oracle advantage is large, then RL has a meaningful switching problem to learn.

Total runtime seconds: 226.651
