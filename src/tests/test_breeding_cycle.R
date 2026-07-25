#########################################################################
# test_breeding_cycle.R
#
# Purpose:
# Test one complete breeding cycle from candidate phenotyping through
# generation of the next candidate population.
#########################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")


#########################################################################
# 1. Load simulator functions
#########################################################################

source("src/simulator/phenotyping.R")
source("src/simulator/genomic_prediction.R")
source("src/simulator/parent_selection.R")
source("src/simulator/next_generation.R")
source("src/simulator/breeding_cycle.R")


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
# 4. Select 200 candidates for this test cycle
#
# Random selection is used only to test the simulator.
#########################################################################

set.seed(12345)

selected_indices <- sample(
    x = seq_len(candidate_population@nInd),
    size = 200,
    replace = FALSE
)


#########################################################################
# 5. Run one complete breeding cycle
#########################################################################

cycle_result <- run_breeding_cycle(
    candidate_population = candidate_population,
    selected_indices = selected_indices,
    generation_number = 1,
    number_of_parents = 20,
    number_of_crosses = 100,
    f1_per_cross = 1,
    dh_per_f1 = 10,
    reps = 1,
    trait = 1,
    seed = 12345,
    simParam = SP
)


#########################################################################
# 6. Extract outputs
#########################################################################

cycle_summary <-
    cycle_result$cycle_summary

next_candidate_population <-
    cycle_result$next_candidate_population


#########################################################################
# 7. Print results
#########################################################################

cat("\nComplete breeding cycle ran successfully.\n")

cat(
    "Current generation:",
    cycle_summary$generation,
    "\n"
)

cat(
    "Current population size:",
    cycle_summary$population_size,
    "\n"
)

cat(
    "Number phenotyped:",
    cycle_summary$number_phenotyped,
    "\n"
)

cat(
    "Prediction accuracy:",
    round(cycle_summary$prediction_accuracy, 3),
    "\n"
)

cat(
    "Number of selected parents:",
    cycle_summary$number_of_selected_parents,
    "\n"
)

cat(
    "Next-generation population size:",
    next_candidate_population@nInd,
    "\n"
)

cat(
    "Realized genetic gain:",
    round(cycle_summary$realized_genetic_gain, 3),
    "\n"
)

cat("\nCycle summary:\n")
print(cycle_summary)


#########################################################################
# 8. Basic tests
#########################################################################

stopifnot(
    cycle_summary$generation == 1
)

stopifnot(
    cycle_summary$population_size == 1000
)

stopifnot(
    cycle_summary$number_phenotyped == 200
)

stopifnot(
    cycle_summary$training_population_size == 200
)

stopifnot(
    cycle_summary$number_of_selected_parents == 20
)

stopifnot(
    cycle_result$selected_parents@nInd == 20
)

stopifnot(
    cycle_result$next_f1_population@nInd == 100
)

stopifnot(
    next_candidate_population@nInd == 1000
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
    is.finite(
        cycle_summary$prediction_accuracy
    )
)

stopifnot(
    is.finite(
        cycle_summary$realized_genetic_gain
    )
)

stopifnot(
    cycle_summary$selected_parent_mean_gv >
    cycle_summary$population_mean_gv_before
)

cat("\nAll complete-breeding-cycle checks passed.\n")