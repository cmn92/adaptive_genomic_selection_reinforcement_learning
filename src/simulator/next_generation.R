#########################################################################
# next_generation.R
#
# Purpose:
# Create the next generation of maize doubled-haploid candidate lines
# from selected breeding parents.
#
# Default breeding structure:
#   - 20 selected parents
#   - 100 random biparental crosses
#   - 1 F1 individual per cross
#   - 10 doubled-haploid lines per F1
#   - 1,000 candidate DH lines in the next generation
#########################################################################


#########################################################################
# 1. Load required package
#########################################################################

library(AlphaSimR)


#########################################################################
# 2. Validate the selected-parent population
#########################################################################

validate_selected_parents <- function(
    selected_parents,
    number_of_crosses
) {

    if (!inherits(selected_parents, "Pop")) {
        stop(
            "'selected_parents' must be an AlphaSimR Pop object."
        )
    }

    if (selected_parents@nInd < 2) {
        stop(
            "At least two selected parents are required for crossing."
        )
    }

    if (
        !is.numeric(number_of_crosses) ||
        length(number_of_crosses) != 1 ||
        number_of_crosses %% 1 != 0
    ) {
        stop(
            "'number_of_crosses' must be one whole number."
        )
    }

    if (number_of_crosses < 1) {
        stop(
            "'number_of_crosses' must be at least 1."
        )
    }

    invisible(TRUE)
}

###########################################################################
# 3. Create the next candidate generation
############################################################################
create_next_generation <- function (
    selected_parents,
    number_of_crosses = 100,
    f1_per_cross = 1,
    dh_per_f1 = 10,
    generation_number = 2,
    seed = NULL,
    simParam = NULL
) {
    validate_selected_parents(
        selected_parents = selected_parents,
        number_of_crosses = number_of_crosses
    )

    #####################################################################
    # Validate F1 progeny number
    #####################################################################

    if (
        !is.numeric(f1_per_cross) ||
        length(f1_per_cross) != 1 ||
        f1_per_cross %% 1 != 0 ||
        f1_per_cross < 1
    ) {
        stop(
            "'f1_per_cross' must be one positive whole number."
        )
    }

    #####################################################################
    # Validate doubled-haploid number
    #####################################################################

    if (
        !is.numeric(dh_per_f1) ||
        length(dh_per_f1) != 1 ||
        dh_per_f1 %% 1 != 0 ||
        dh_per_f1 < 1
    ) {
        stop(
            "'dh_per_f1' must be one positive whole number."
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
    # Set random seed if supplied
    #####################################################################

    if (!is.null(seed)) {
        set.seed(seed)
    }


    #####################################################################
    # Create random biparental crosses
    #####################################################################

    f1_population <- randCross(
        pop = selected_parents,
        nCrosses = as.integer(number_of_crosses),
        nProgeny = as.integer(f1_per_cross),
        balance = TRUE,
        simParam = simParam
    )

    #####################################################################
    # Create doubled-haploid candidate lines
    #
    # makeDH creates dh_per_f1 DH lines from each F1 individual.
    #####################################################################

    next_candidate_population <- makeDH(
        pop = f1_population,
        nDH = as.integer(dh_per_f1),
        simParam = simParam
    )

    #####################################################################
    # Calculate expected and observed population sizes
    #####################################################################

    expected_f1_size <-
        number_of_crosses * f1_per_cross

    expected_candidate_size <-
        expected_f1_size * dh_per_f1

    observed_f1_size <-
        f1_population@nInd

    observed_candidate_size <-
        next_candidate_population@nInd

    #####################################################################
    # Confirm that population sizes are correct
    #####################################################################

    if (observed_f1_size != expected_f1_size) {
        stop(
            paste0(
                "Unexpected F1 population size. Expected ",
                expected_f1_size,
                " but obtained ",
                observed_f1_size,
                "."
            )
        )
    }

    if (observed_candidate_size != expected_candidate_size) {
        stop(
            paste0(
                "Unexpected candidate population size. Expected ",
                expected_candidate_size,
                " but obtained ",
                observed_candidate_size,
                "."
            )
        )
    }

    #####################################################################
    # Assign candidate identifiers
    #
    # Example:
    # G02_CAND_0001
    #####################################################################

    next_candidate_population@id <- sprintf(
        "G%02d_CAND_%04d",
        as.integer(generation_number),
        seq_len(next_candidate_population@nInd)
    )

    #####################################################################
    # Create crossing summary
    #####################################################################

    crossing_summary <- data.frame(
        generation = as.integer(generation_number),
        number_of_parents = selected_parents@nInd,
        number_of_crosses = as.integer(number_of_crosses),
        f1_per_cross = as.integer(f1_per_cross),
        number_of_f1 = observed_f1_size,
        dh_per_f1 = as.integer(dh_per_f1),
        number_of_candidates = observed_candidate_size,
        stringsAsFactors = FALSE
    )


    #####################################################################
    # Return useful outputs
    #####################################################################

    return(
        list(
            selected_parents = selected_parents,
            f1_population = f1_population,
            candidate_population = next_candidate_population,
            crossing_summary = crossing_summary
        )
    )

}