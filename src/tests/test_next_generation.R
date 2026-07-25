#########################################################################
# test_next_generation.R
#
# Purpose:
# Test the full sequence from phenotyping to creation of the next
# candidate generation.
#########################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")


#########################################################################
# 1. Load project functions
#########################################################################

source("src/simulator/phenotyping.R")
source("src/simulator/genomic_prediction.R")
source("src/simulator/parent_selection.R")
source("src/simulator/next_generation.R")


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
    stop(
        "'candidate_population' was not found."
    )
}

if (!exists("SP")) {
    stop(
        "'SP' was not found."
    )
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

selected_parents <-
    selection_result$selected_parents


#########################################################################
# 7. Create the next candidate generation
#########################################################################

next_generation_result <- create_next_generation(
    selected_parents = selected_parents,
    number_of_crosses = 100,
    f1_per_cross = 1,
    dh_per_f1 = 10,
    generation_number = 2,
    seed = 12345,
    simParam = SP
)

next_f1_population <-
    next_generation_result$f1_population

next_candidate_population <-
    next_generation_result$candidate_population

crossing_summary <-
    next_generation_result$crossing_summary


#########################################################################
# 8. Print results
#########################################################################

cat("\nNext generation created successfully.\n")

cat(
    "Number of selected parents:",
    selected_parents@nInd,
    "\n"
)

cat(
    "Number of F1 individuals:",
    next_f1_population@nInd,
    "\n"
)

cat(
    "Number of next-generation candidates:",
    next_candidate_population@nInd,
    "\n"
)

cat("\nCrossing summary:\n")
print(crossing_summary)

cat("\nFirst six candidate IDs:\n")
print(
    head(next_candidate_population@id)
)


#########################################################################
# 9. Basic tests
#########################################################################

stopifnot(
    selected_parents@nInd == 20
)

stopifnot(
    next_f1_population@nInd == 100
)

stopifnot(
    next_candidate_population@nInd == 1000
)

stopifnot(
    length(unique(next_candidate_population@id)) == 1000
)

stopifnot(
    all(
        grepl(
            "^G02_CAND_[0-9]{4}$",
            next_candidate_population@id
        )
    )
)

stopifnot(
    nrow(crossing_summary) == 1
)

stopifnot(
    crossing_summary$number_of_candidates == 1000
)

cat("\nAll next-generation checks passed.\n")