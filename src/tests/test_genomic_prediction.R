#########################################################################
# test_genomic_prediction.R
#
# Purpose:
# Test RR-BLUP genomic prediction using the phenotyped candidate subset.
#########################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")


#########################################################################
# 1. Load project functions
#########################################################################

source("src/simulator/phenotyping.R")
source("src/simulator/genomic_prediction.R")


#########################################################################
# 2. Load the initial candidate population
#########################################################################

input_file <- "data/initial_candidate_population.RData"

if (!file.exists(input_file)) {
    stop(
        paste(
            "Candidate population file does not exist:",
            input_file
        )
    )
}

loaded_objects <- load(input_file)

cat("\nObjects loaded:\n")
print(loaded_objects)


#########################################################################
# 3. Check required objects
#########################################################################

if (!exists("candidate_population")) {
    stop("'candidate_population' was not found.")
}

if (!exists("SP")) {
    stop("'SP' was not found.")
}


#########################################################################
# 4. Randomly phenotype 200 candidates
#########################################################################

phenotyping_result <- phenotype_random_sample(
    population = candidate_population,
    number_to_phenotype = 200,
    reps = 1,
    seed = 12345,
    simParam = SP
)

training_population <-
    phenotyping_result$phenotyped_population


#########################################################################
# 5. Fit RR-BLUP and predict all 1,000 candidates
#########################################################################

prediction_result <- fit_rrblup_prediction(
    training_population = training_population,
    candidate_population = candidate_population,
    trait = 1,
    simParam = SP
)


#########################################################################
# 6. Inspect results
#########################################################################

prediction_table <- prediction_result$prediction_table
predicted_population <- prediction_result$predicted_population
prediction_accuracy <- prediction_result$prediction_accuracy

cat("\nGenomic prediction completed successfully.\n")
cat("Training population size:", training_population@nInd, "\n")
cat("Candidate population size:", predicted_population@nInd, "\n")
cat(
    "Prediction accuracy:",
    round(prediction_accuracy, 3),
    "\n"
)

cat("\nFirst six prediction records:\n")
print(head(prediction_table))


#########################################################################
# 7. Basic tests
#########################################################################

stopifnot(
    nrow(prediction_table) == candidate_population@nInd
)

stopifnot(
    all(!is.na(prediction_table$predicted_gebv))
)

stopifnot(
    length(unique(prediction_table$individual_id)) ==
        candidate_population@nInd
)

stopifnot(
    is.numeric(prediction_accuracy)
)

stopifnot(
    prediction_accuracy >= -1 &&
    prediction_accuracy <= 1
)

cat("\nAll genomic prediction checks passed.\n")