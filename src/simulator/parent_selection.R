#####################################################################################
# parent_selection.R
#
# Purpose:
# Select breeding parents from a candidate population using GEBVs
#
# The first selection method is truncation selection:
# choose the individuals with the highest GEBVs as parents for the next generation.

#############################################################################
# 1. Load required package
#############################################################################

library(AlphaSimR)

#############################################################################
# 2. Validate the predicted population
#############################################################################

validate_selection_population <- function(
    predicted_population,
    trait = 1
) {

    if (!inherits(predicted_population, "Pop")) {
        stop(
            "'predicted_population' must be an AlphaSimR Pop object."
        )
    }

    estimated_values <- ebv(predicted_population)

    if (is.null(estimated_values)) {
        stop(
            paste(
                "The population does not contain estimated breeding values.",
                "Run genomic prediction before parent selection."
            )
        )
    }

    if (nrow(estimated_values) != predicted_population@nInd) {
        stop(
            "The number of EBVs does not match the population size."
        )
    }

    if (
        !is.numeric(trait) ||
        length(trait) != 1 ||
        trait %% 1 != 0 ||
        trait < 1
    ) {
        stop("'trait' must be one positive whole number.")
    }

    if (trait > ncol(estimated_values)) {
        stop(
            "'trait' exceeds the number of available EBV traits."
        )
    }

    if (anyNA(estimated_values[, trait])) {
        stop(
            "The selected trait contains missing estimated breeding values."
        )
    }

    invisible(TRUE)
}


#############################################################################
# 3. Select the top parents by GEBV
#############################################################################

select_top_parents <- function(predicted_population, number_of_parents, trait=1){
    validate_selection_population(
        predicted_population= predicted_population,
        trait = trait
    )
    if (
        !is.numeric(number_of_parents) ||
        length(number_of_parents) != 1 ||
        number_of_parents %% 1 != 0
    ) {
        stop("'number_of_parents' must be one whole number.")
    }

    if (number_of_parents < 2) {
        stop(
            "At least two parents are required for crossing."
        )
    }

    if (number_of_parents > predicted_population@nInd) {
        stop(
            paste0(
                "'number_of_parents' cannot exceed population size ",
                predicted_population@nInd,
                "."
            )
        )
    }

    #####################################################################
    # Extract GEBVs for the chosen trait
    #####################################################################

    predicted_gebv <- as.numeric(
        ebv(predicted_population)[, trait]
    )

    #####################################################################
    # Rank candidates from highest to lowest GEBV
    #####################################################################

    ranked_indices <- order(
        predicted_gebv,
        decreasing = TRUE
    )

    selected_indices <- ranked_indices[
        seq_len(number_of_parents)
    ]

    #####################################################################
    # Extract selected parents as a new AlphaSimR population
    #####################################################################

    selected_parents <- predicted_population[
        selected_indices
    ]

    #####################################################################
    # Create a selection-results table
    #####################################################################

    selected_true_bv <- as.numeric(
        bv(predicted_population)[selected_indices, trait]
    )

    selection_table <- data.frame(
        selection_rank = seq_len(number_of_parents),
        population_index = selected_indices,
        individual_id = predicted_population@id[selected_indices],
        predicted_gebv = predicted_gebv[selected_indices],
        true_breeding_value = selected_true_bv,
        stringsAsFactors = FALSE
    )

    selection_table$prediction_error <-
        selection_table$predicted_gebv -
        selection_table$true_breeding_value


    #####################################################################
    # Return useful outputs
    #####################################################################

    return(
        list(
            selected_parents = selected_parents,
            selected_indices = selected_indices,
            selection_table = selection_table
        )
    )


}