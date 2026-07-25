#########################################################################
# test_run_generations.R
#
# Purpose:
# Test whether the breeding simulator can run continuously across
# several generations using random phenotyping.
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
source("src/simulator/run_generations.R")


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
# 4. Run five generations
#
# Five generations are enough for the first stress test.
# Later, the main experiments can use 10, 15, or 20 generations.
#########################################################################

simulation_result <- run_generations(
    initial_population = candidate_population,
    number_of_generations = 5,
    number_to_phenotype = 200,
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
# 5. Extract outputs
#########################################################################

simulation_summary <-
    simulation_result$simulation_summary

overall_summary <-
    simulation_result$overall_summary

final_population <-
    simulation_result$final_population


#########################################################################
# 6. Print results
#########################################################################

cat("\nMulti-generation simulation completed successfully.\n")

cat("\nGeneration-by-generation summary:\n")
print(simulation_summary)

cat("\nOverall simulation summary:\n")
print(overall_summary)

cat(
    "\nFinal population size:",
    final_population@nInd,
    "\n"
)


#########################################################################
# 7. Basic tests
#########################################################################

stopifnot(
    nrow(simulation_summary) == 5
)

stopifnot(
    all(simulation_summary$generation == 1:5)
)

stopifnot(
    all(simulation_summary$population_size == 1000)
)

stopifnot(
    all(simulation_summary$number_phenotyped == 200)
)

stopifnot(
    all(simulation_summary$training_population_size == 200)
)

stopifnot(
    all(simulation_summary$number_of_selected_parents == 20)
)

stopifnot(
    final_population@nInd == 1000
)

stopifnot(
    length(simulation_result$cycle_results) == 5
)

stopifnot(
    all(
        is.finite(
            simulation_summary$prediction_accuracy
        )
    )
)

stopifnot(
    all(
        is.finite(
            simulation_summary$realized_genetic_gain
        )
    )
)

stopifnot(
    overall_summary$total_number_phenotyped == 1000
)

stopifnot(
    overall_summary$cumulative_phenotyping_cost == 1000
)

stopifnot(
    isTRUE(
        all.equal(
            overall_summary$final_mean_genetic_value,
            simulation_summary$next_generation_mean_gv[5]
        )
    )
)

cat("\nAll multi-generation checks passed.\n")