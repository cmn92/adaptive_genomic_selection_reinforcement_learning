############################################################################
# breeding_cycle.R
#
# Purpose:
# Run one complete genomic-selection breeding cycle
#
# The cycle performs:
# 1. Select candidates for phenotyping
# 2. Generate phenotypes for the selected candidates
# 3. Fit an RR-BLUP genomic prediction model
# 4. Predict GEBVs for all candidates
# 5. Select the top parents by GEBV
# 6. Cross selected parents
# 7. Generate the next DH candidate population
# 8. Calculate and return breeding-cycle metrics

#########################################################################
# 1. Load required package
#########################################################################

library(AlphaSimR)


#########################################################################
# 2. Run one complete breeding cycle
#########################################################################

run_breeding_cycle <- function(
    candidate_population,
    selected_indices,
    generation_number,
    number_of_parents = 20,
    number_of_crosses = 100,
    f1_per_cross = 1,
    dh_per_f1 = 10,
    reps = 1,
    trait = 1,
    seed = NULL,
    simParam = NULL
) {

    #####################################################################
    # Validate candidate population
    #####################################################################

    if (!inherits(candidate_population, "Pop")) {
        stop(
            "'candidate_population' must be an AlphaSimR Pop object."
        )
    }


    #####################################################################
    # Validate generation number
    #####################################################################

    if (
        !is.numeric(generation_number) ||
        length(generation_number) != 1 ||
        generation_number %% 1 != 0 ||
        generation_number < 1
    ) {
        stop(
            "'generation_number' must be one positive whole number."
        )
    }


    #####################################################################
    # Retrieve SimParam object if not supplied
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
    # Set random seed
    #####################################################################

    if (!is.null(seed)) {
        set.seed(seed)
    }


    #####################################################################
    # Record the population before selection
    #####################################################################

    current_true_gv <- as.numeric(
        gv(candidate_population)[, trait]
    )

    population_mean_gv_before <- mean(
        current_true_gv,
        na.rm = TRUE
    )

    population_variance_gv_before <- var(
        current_true_gv,
        na.rm = TRUE
    )


    #####################################################################
    # 1. Phenotype the selected candidates
    #####################################################################

    phenotyping_result <- phenotype_selected(
        population = candidate_population,
        selected_indices = selected_indices,
        reps = reps,
        simParam = simParam
    )

    training_population <-
        phenotyping_result$phenotyped_population

    phenotype_table <-
        phenotyping_result$phenotype_table


    #####################################################################
    # 2. Fit RR-BLUP and predict all candidates
    #####################################################################

    prediction_result <- fit_rrblup_prediction(
        training_population = training_population,
        candidate_population = candidate_population,
        trait = trait,
        simParam = simParam
    )

    predicted_population <-
        prediction_result$predicted_population

    prediction_table <-
        prediction_result$prediction_table

    prediction_accuracy <-
        prediction_result$prediction_accuracy


    #####################################################################
    # 3. Select the top parents by GEBV
    #####################################################################

    selection_result <- select_top_parents(
        predicted_population = predicted_population,
        number_of_parents = number_of_parents,
        trait = trait
    )

    selected_parents <-
        selection_result$selected_parents

    selection_table <-
        selection_result$selection_table


    #####################################################################
    # 4. Record selected-parent metrics
    #####################################################################

    selected_indices_parent <-
        selection_result$selected_indices

    selected_parent_true_gv <- as.numeric(
        gv(predicted_population)[
            selected_indices_parent,
            trait
        ]
    )

    selected_parent_mean_gv <- mean(
        selected_parent_true_gv,
        na.rm = TRUE
    )

    selection_differential <-
        selected_parent_mean_gv -
        population_mean_gv_before


    #####################################################################
    # 5. Create the next candidate generation
    #####################################################################

    next_generation_result <- create_next_generation(
        selected_parents = selected_parents,
        number_of_crosses = number_of_crosses,
        f1_per_cross = f1_per_cross,
        dh_per_f1 = dh_per_f1,
        generation_number = generation_number + 1,
        seed = seed,
        simParam = simParam
    )

    next_candidate_population <-
        next_generation_result$candidate_population

    crossing_summary <-
        next_generation_result$crossing_summary


    #####################################################################
    # 6. Record next-generation metrics
    #####################################################################

    next_generation_true_gv <- as.numeric(
        gv(next_candidate_population)[, trait]
    )

    next_generation_mean_gv <- mean(
        next_generation_true_gv,
        na.rm = TRUE
    )

    next_generation_variance_gv <- var(
        next_generation_true_gv,
        na.rm = TRUE
    )

    realized_genetic_gain <-
        next_generation_mean_gv -
        population_mean_gv_before


    #####################################################################
    # 7. Calculate phenotyping cost
    #
    # For now, one phenotype record costs one unit.
    # We will later replace this with an explicit economic cost.
    #####################################################################

    number_phenotyped <- length(
        selected_indices
    )

    phenotyping_cost_units <-
        number_phenotyped * reps


    #####################################################################
    # 8. Create cycle summary
    #####################################################################

    cycle_summary <- data.frame(
        generation = as.integer(generation_number),

        population_size =
            candidate_population@nInd,

        number_phenotyped =
            number_phenotyped,

        phenotyping_fraction =
            number_phenotyped /
            candidate_population@nInd,

        replications =
            as.integer(reps),

        phenotyping_cost_units =
            phenotyping_cost_units,

        training_population_size =
            training_population@nInd,

        prediction_accuracy =
            prediction_accuracy,

        number_of_selected_parents =
            selected_parents@nInd,

        population_mean_gv_before =
            population_mean_gv_before,

        population_variance_gv_before =
            population_variance_gv_before,

        selected_parent_mean_gv =
            selected_parent_mean_gv,

        selection_differential =
            selection_differential,

        next_generation_mean_gv =
            next_generation_mean_gv,

        next_generation_variance_gv =
            next_generation_variance_gv,

        realized_genetic_gain =
            realized_genetic_gain,

        stringsAsFactors = FALSE
    )


    #####################################################################
    # 9. Return all useful cycle outputs
    #####################################################################

    return(
        list(
            current_population =
                candidate_population,

            selected_indices =
                selected_indices,

            training_population =
                training_population,

            phenotype_table =
                phenotype_table,

            rrblup_model =
                prediction_result$model,

            predicted_population =
                predicted_population,

            prediction_table =
                prediction_table,

            prediction_accuracy =
                prediction_accuracy,

            selected_parents =
                selected_parents,

            selection_table =
                selection_table,

            next_f1_population =
                next_generation_result$f1_population,

            next_candidate_population =
                next_candidate_population,

            crossing_summary =
                crossing_summary,

            cycle_summary =
                cycle_summary
        )
    )
}