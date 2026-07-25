#########################################################################
# test_parent_selection.R
#
# Purpose:
# Test parent selection after phenotyping and genomic prediction.
#########################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")


#########################################################################
# 1. Load project functions
#########################################################################

source("src/simulator/phenotyping.R")
source("src/simulator/genomic_prediction.R")
source("src/simulator/parent_selection.R")


#########################################################################
# 2. Load candidate population
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
# 4. Phenotype 200 candidates
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
# 5. Fit RR-BLUP and predict all candidates
#########################################################################

prediction_result <- fit_rrblup_prediction(
    training_population = training_population,
    candidate_population = candidate_population,
    trait = 1,
    simParam = SP
)

predicted_population <-
    prediction_result$predicted_population


#########################################################################
# 6. Select the top 20 parents
#########################################################################

selection_result <- select_top_parents(
    predicted_population = predicted_population,
    number_of_parents = 20,
    trait = 1
)

selected_parents <- selection_result$selected_parents
selection_table <- selection_result$selection_table


#########################################################################
# 7. Inspect results
#########################################################################

cat("\nParent selection completed successfully.\n")
cat(
    "Candidate population size:",
    candidate_population@nInd,
    "\n"
)

cat(
    "Number of selected parents:",
    selected_parents@nInd,
    "\n"
)

cat("\nSelected parents:\n")
print(selection_table)


#########################################################################
# 8. Basic tests
#########################################################################

stopifnot(
    selected_parents@nInd == 20
)

stopifnot(
    nrow(selection_table) == 20
)

stopifnot(
    length(unique(selection_table$individual_id)) == 20
)

stopifnot(
    all(
        diff(selection_table$predicted_gebv) <= 0
    )
)

all_candidate_gebvs <- as.numeric(
    ebv(predicted_population)[, 1]
)

expected_top_gebvs <- sort(
    all_candidate_gebvs,
    decreasing = TRUE
)[seq_len(20)]

stopifnot(
    isTRUE(
        all.equal(
            selection_table$predicted_gebv,
            expected_top_gebvs
        )
    )
)

cat("\nAll parent-selection checks passed.\n")