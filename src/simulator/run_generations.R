#########################################################################
# run_generations.R
#
# Purpose:
# Run the genomic-selection simulator across multiple breeding
# generations.
#
# For this first version, candidates are selected for phenotyping
# randomly in every generation.
#
# Later, the random selection step will be replaced by:
#   - fixed phenotyping
#   - diversity-based phenotyping
#   - active learning
#   - reinforcement learning
#########################################################################


#########################################################################
# 1. Load required package
#########################################################################

library(AlphaSimR)


#########################################################################
# 2. Validate multi-generation simulation inputs
#########################################################################

validate_generation_inputs <- function(
    initial_population,
    number_of_generations,
    number_to_phenotype
) {

    if (!inherits(initial_population, "Pop")) {
        stop(
            "'initial_population' must be an AlphaSimR Pop object."
        )
    }

    if (
        !is.numeric(number_of_generations) ||
        length(number_of_generations) != 1 ||
        number_of_generations %% 1 != 0 ||
        number_of_generations < 1
    ) {
        stop(
            "'number_of_generations' must be one positive whole number."
        )
    }

    if (
        !is.numeric(number_to_phenotype) ||
        length(number_to_phenotype) != 1 ||
        number_to_phenotype %% 1 != 0 ||
        number_to_phenotype < 1
    ) {
        stop(
            "'number_to_phenotype' must be one positive whole number."
        )
    }

    if (number_to_phenotype > initial_population@nInd) {
        stop(
            paste0(
                "'number_to_phenotype' cannot exceed the population size of ",
                initial_population@nInd,
                "."
            )
        )
    }

    invisible(TRUE)
}


#########################################################################
# 3. Run multiple breeding generations
#########################################################################

run_generations <- function(
    initial_population,
    number_of_generations = 10,
    number_to_phenotype = 200,
    number_of_parents = 20,
    number_of_crosses = 100,
    f1_per_cross = 1,
    dh_per_f1 = 10,
    reps = 1,
    trait = 1,
    seed = 12345,
    simParam = NULL
) {

    validate_generation_inputs(
        initial_population = initial_population,
        number_of_generations = number_of_generations,
        number_to_phenotype = number_to_phenotype
    )


    #####################################################################
    # Retrieve SimParam object if one was not supplied
    #####################################################################

    if (is.null(simParam)) {

        if (!exists("SP", envir = .GlobalEnv)) {
            stop(
                paste(
                    "No SimParam object was supplied and no global",
                    "object named 'SP' was found."
                )
            )
        }

        simParam <- get(
            "SP",
            envir = .GlobalEnv
        )
    }


    #####################################################################
    # Set the main simulation seed
    #####################################################################

    if (!is.null(seed)) {
        set.seed(seed)
    }


    #####################################################################
    # Initialize storage objects
    #####################################################################

    current_population <- initial_population

    cycle_results <- vector(
        mode = "list",
        length = number_of_generations
    )

    generation_summaries <- vector(
        mode = "list",
        length = number_of_generations
    )

    cumulative_phenotyping_cost <- 0


    #####################################################################
    # Run each breeding generation
    #####################################################################

    for (generation in seq_len(number_of_generations)) {

        cat(
            "\nRunning generation",
            generation,
            "of",
            number_of_generations,
            "...\n"
        )


        #################################################################
        # Use a distinct but reproducible seed for this generation
        #################################################################

        generation_seed <- NULL

        if (!is.null(seed)) {
            generation_seed <- seed + generation - 1
            set.seed(generation_seed)
        }


        #################################################################
        # Randomly select candidates for phenotyping
        #
        # This is temporary. Later, a phenotyping strategy will supply
        # these indices.
        #################################################################

        selected_indices <- sample(
            x = seq_len(current_population@nInd),
            size = number_to_phenotype,
            replace = FALSE
        )


        #################################################################
        # Run one complete breeding cycle
        #################################################################

        cycle_result <- run_breeding_cycle(
            candidate_population = current_population,
            selected_indices = selected_indices,
            generation_number = generation,
            number_of_parents = number_of_parents,
            number_of_crosses = number_of_crosses,
            f1_per_cross = f1_per_cross,
            dh_per_f1 = dh_per_f1,
            reps = reps,
            trait = trait,
            seed = generation_seed,
            simParam = simParam
        )


        #################################################################
        # Update cumulative cost
        #################################################################

        generation_cost <-
            cycle_result$cycle_summary$phenotyping_cost_units

        cumulative_phenotyping_cost <-
            cumulative_phenotyping_cost +
            generation_cost


        #################################################################
        # Add multi-generation information to the summary
        #################################################################

        generation_summary <- cycle_result$cycle_summary

        generation_summary$cumulative_phenotyping_cost <-
            cumulative_phenotyping_cost

        generation_summary$population_id_prefix <-
            paste0(
                "G",
                sprintf("%02d", generation)
            )


        #################################################################
        # Store outputs
        #################################################################

        cycle_results[[generation]] <- cycle_result
        generation_summaries[[generation]] <- generation_summary


        #################################################################
        # Print a compact generation summary
        #################################################################

        cat(
            "Prediction accuracy:",
            round(generation_summary$prediction_accuracy, 3),
            "\n"
        )

        cat(
            "Current mean genetic value:",
            round(generation_summary$population_mean_gv_before, 3),
            "\n"
        )

        cat(
            "Next-generation mean genetic value:",
            round(generation_summary$next_generation_mean_gv, 3),
            "\n"
        )

        cat(
            "Realized genetic gain:",
            round(generation_summary$realized_genetic_gain, 3),
            "\n"
        )

        cat(
            "Genetic variance:",
            round(generation_summary$next_generation_variance_gv, 3),
            "\n"
        )


        #################################################################
        # Move to the next generation
        #################################################################

        current_population <-
            cycle_result$next_candidate_population
    }


    #####################################################################
    # Combine all summaries into one data frame
    #####################################################################

    simulation_summary <- do.call(
        rbind,
        generation_summaries
    )

    rownames(simulation_summary) <- NULL


    #####################################################################
    # Calculate additional long-term metrics
    #####################################################################

    initial_mean_gv <-
        simulation_summary$population_mean_gv_before[1]

    final_mean_gv <-
        simulation_summary$next_generation_mean_gv[
            nrow(simulation_summary)
        ]

    total_realized_gain <-
        final_mean_gv -
        initial_mean_gv

    mean_prediction_accuracy <-
        mean(
            simulation_summary$prediction_accuracy,
            na.rm = TRUE
        )

    final_genetic_variance <-
        simulation_summary$next_generation_variance_gv[
            nrow(simulation_summary)
        ]


    #####################################################################
    # Create an overall simulation summary
    #####################################################################

    overall_summary <- data.frame(
        number_of_generations =
            number_of_generations,

        number_phenotyped_per_generation =
            number_to_phenotype,

        total_number_phenotyped =
            number_of_generations *
            number_to_phenotype,

        cumulative_phenotyping_cost =
            cumulative_phenotyping_cost,

        initial_mean_genetic_value =
            initial_mean_gv,

        final_mean_genetic_value =
            final_mean_gv,

        total_realized_genetic_gain =
            total_realized_gain,

        mean_prediction_accuracy =
            mean_prediction_accuracy,

        final_genetic_variance =
            final_genetic_variance,

        stringsAsFactors = FALSE
    )


    #####################################################################
    # Return all useful outputs
    #####################################################################

    return(
        list(
            initial_population =
                initial_population,

            final_population =
                current_population,

            cycle_results =
                cycle_results,

            simulation_summary =
                simulation_summary,

            overall_summary =
                overall_summary
        )
    )
}