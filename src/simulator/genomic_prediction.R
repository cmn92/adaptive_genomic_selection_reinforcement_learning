#############################################################################
# genomic_prediction.R
#
# Purpose:
# Fit an RR-BLUP genomic prediction model using phenotyped candidate lines and
# predict genomic estimated breeding values (GEBVs) for all candidate lines.
#############################################################################


#############################################################################
# 1. Load required libraries
#############################################################################

library(AlphaSimR)

##############################################################################
# 2. Validate the training and candidate populations
##############################################################################

validate_prediction_inputs <- function(training_population, candidate_population) {
    if (!inherits(training_population, "Pop")) {
        stop("The provided training_population is not a valid AlphaSimR Pop object.")
    }

    if (!inherits(candidate_population, "Pop")) {
        stop("The provided candidate_population is not a valid AlphaSimR Pop object.")
    }

    if (training_population@nInd < 2) {
        stop("The training population must contain at least two individuals for genomic prediction.")
    }

    if (all(is.na(training_population@pheno))) {
        stop("The training population does not have any phenotypic data. Please ensure that phenotypes are available for genomic prediction.")
    }

    invisible(TRUE)
}

###############################################################################
# 3. Fit RR-BLUP model and predict GEBVs
################################################################################

fit_rrblup_prediction <- function(
    training_population,
    candidate_population,
    trait = 1,
    simParam = NULL
) {
    validate_prediction_inputs(training_population, candidate_population)

    if (
        !is.numeric(trait) ||
        length(trait) != 1 ||
        trait < 1 ||
        trait %% 1 != 0 
    ) {
        stop("The 'trait' argument must be a single positive integer corresponding to the trait index.")
    }

    if (trait > ncol(training_population@pheno)) {
        stop("The specified trait index exceeds the number of traits in the training population.")
    }

    if (is.null(simParam)) {

        if (!exists("SP", envir = .GlobalEnv)) {
            stop("The 'simParam' argument is NULL and no global 'SP' object is found. Please provide a valid simParam object.")
        }
        simParam <- get("SP", envir = .GlobalEnv)
    }

    #########################################################################
    # Fit the RR-BLUP model using the phenotyped training population
    #########################################################################

    rrblup_model <- RRBLUP(
        pop = training_population,
        traits = trait,
        use = "pheno",
        snpChip = 1,
        simParam = simParam
    )


    #########################################################################
    # Apply the fitted model to all candidate individuals
    #########################################################################

    predicted_population <- setEBV(
        pop = candidate_population,
        solution = rrblup_model,
        value = "bv",
        simParam = simParam
    )

    ###########################################################################
    # Extract predicted and true values
    ###########################################################################

    predicted_gebv <- ebv(predicted_population)[, 1]

    true_breeding_value <- bv(predicted_population)[, trait]


    ###########################################################################
    # Calculate prediction accuracy
    #
    # This is the correlation between predicted breeding values and simulated true
    # genetic values for the specified trait.
    ###########################################################################

    prediction_accuracy <- cor(predicted_gebv, true_breeding_value, use = "complete.obs")

    ###########################################################################
    # Create a clean prediction table
    ###########################################################################

    prediction_table <- data.frame(
        population_index = seq_len(predicted_population@nInd),
        individual_id = predicted_population@id,
        predicted_gebv = predicted_gebv,
        true_breeding_value = true_breeding_value,
        stringsAsFactors = FALSE
    )

    prediction_table$prediction_error <- prediction_table$predicted_gebv - prediction_table$true_breeding_value

    ##########################################################################
    # Return useful outputs
    ##########################################################################

    return(
        list(
            model = rrblup_model,
            predicted_population = predicted_population,
            prediction_table = prediction_table,
            prediction_accuracy = prediction_accuracy
        )
    )
}