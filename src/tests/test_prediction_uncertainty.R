###############################################################################
# test_prediction_uncertainty.R
###############################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")

source("src/simulator/phenotyping.R")
source("src/simulator/prediction_uncertainty.R")

load("data/initial_candidate_population.RData")

phenotyping_result <- phenotype_random_sample(
    population = candidate_population,
    number_to_phenotype = 50,
    reps = 1,
    seed = 12345,
    simParam = SP
)

training_population <- phenotyping_result$phenotyped_population

uncertainty_result <- compute_prediction_uncertainty(
    candidate_population = candidate_population,
    training_population = training_population,
    trait = 1,
    snp_chip = 1,
    n_cores = 1,
    simParam = SP
)

uncertainty_table <- uncertainty_result$uncertainty_table
unphenotyped_table <- uncertainty_result$unphenotyped_uncertainty

selected_second_batch <- select_highest_pev_candidates(
    uncertainty_result = uncertainty_result,
    number_to_select = 150
)

cat("\nPrediction uncertainty calculated successfully.\n")
cat(
    "Candidate population size:",
    uncertainty_result$candidate_population_size,
    "\n"
)
cat(
    "Training population size:",
    uncertainty_result$training_population_size,
    "\n"
)
cat(
    "Unphenotyped candidates:",
    nrow(unphenotyped_table),
    "\n"
)
cat(
    "Second-batch candidates selected:",
    nrow(selected_second_batch),
    "\n"
)

cat("\nMost uncertain candidates:\n")
print(
    head(
        selected_second_batch[
            ,
            c(
                "population_index",
                "individual_id",
                "prediction_error_variance",
                "reliability"
            )
        ]
    )
)

stopifnot(
    nrow(uncertainty_table) == 1000
)

stopifnot(
    sum(uncertainty_table$is_phenotyped) == 50
)

stopifnot(
    nrow(unphenotyped_table) == 950
)

stopifnot(
    nrow(selected_second_batch) == 150
)

stopifnot(
    all(
        is.finite(
            uncertainty_table$prediction_error_variance
        )
    )
)

stopifnot(
    all(
        uncertainty_table$prediction_error_variance >= 0
    )
)

stopifnot(
    all(
        diff(
            unphenotyped_table$prediction_error_variance
        ) <= 0
    )
)

stopifnot(
    !any(
        selected_second_batch$is_phenotyped
    )
)

stopifnot(
    length(
        unique(
            selected_second_batch$individual_id
        )
    ) == 150
)

cat("\nAll prediction-uncertainty checks passed.\n")