###############################################################################
# prediction_uncertainty.R
#
# Purpose:
# Calculate candidate-level genomic prediction uncertainty for active
# phenotyping using G-BLUP and prediction error variance (PEV).
#
# The module:
# 1. Extracts SNP genotypes for all current candidates.
# 2. Converts AlphaSimR dosage coding {0, 1, 2} to rrBLUP coding {-1, 0, 1}.
# 3. Constructs a genomic additive relationship matrix using rrBLUP::A.mat().
# 4. Fits G-BLUP using phenotypes from the current training population.
# 5. Returns predicted breeding values, PEV, and reliability for every
#    candidate.
#
# Active learning will prioritize unphenotyped candidates with the largest PEV.
###############################################################################


###############################################################################
# 1. Check required packages
###############################################################################

if (!requireNamespace("AlphaSimR", quietly = TRUE)) {
    stop(
        "The AlphaSimR package is required but is not installed."
    )
}

if (!requireNamespace("rrBLUP", quietly = TRUE)) {
    stop(
        paste(
            "The rrBLUP package is required but is not installed.",
            "Install it with:",
            'install.packages("rrBLUP")'
        )
    )
}


###############################################################################
# 2. Validate prediction-uncertainty inputs
###############################################################################

validate_uncertainty_inputs <- function(
    candidate_population,
    training_population,
    trait = 1,
    snp_chip = 1
) {

    if (!inherits(candidate_population, "Pop")) {
        stop(
            "'candidate_population' must be an AlphaSimR Pop object."
        )
    }

    if (!inherits(training_population, "Pop")) {
        stop(
            "'training_population' must be an AlphaSimR Pop object."
        )
    }

    if (candidate_population@nInd < 2) {
        stop(
            "The candidate population must contain at least two individuals."
        )
    }

    if (training_population@nInd < 2) {
        stop(
            "The training population must contain at least two individuals."
        )
    }

    if (
        !is.numeric(trait) ||
        length(trait) != 1 ||
        trait %% 1 != 0 ||
        trait < 1
    ) {
        stop(
            "'trait' must be one positive whole number."
        )
    }

    if (
        !is.numeric(snp_chip) ||
        length(snp_chip) != 1 ||
        snp_chip %% 1 != 0 ||
        snp_chip < 1
    ) {
        stop(
            "'snp_chip' must be one positive whole number."
        )
    }

    training_phenotypes <- AlphaSimR::pheno(
        training_population
    )

    if (trait > ncol(training_phenotypes)) {
        stop(
            "'trait' exceeds the number of traits in the training population."
        )
    }

    if (anyNA(training_phenotypes[, trait])) {
        stop(
            "The selected training-population trait contains missing values."
        )
    }

    candidate_ids <- as.character(
        candidate_population@id
    )

    training_ids <- as.character(
        training_population@id
    )

    if (anyDuplicated(candidate_ids)) {
        stop(
            "Candidate-population IDs must be unique."
        )
    }

    if (anyDuplicated(training_ids)) {
        stop(
            "Training-population IDs must be unique."
        )
    }

    missing_training_ids <- setdiff(
        training_ids,
        candidate_ids
    )

    if (length(missing_training_ids) > 0) {
        stop(
            paste0(
                "The following training individuals are not present in ",
                "the candidate population: ",
                paste(
                    head(missing_training_ids, 10),
                    collapse = ", "
                )
            )
        )
    }

    invisible(TRUE)
}


###############################################################################
# 3. Extract and prepare the candidate marker matrix
###############################################################################

prepare_candidate_markers <- function(
    candidate_population,
    snp_chip = 1,
    simParam = NULL
) {

    if (is.null(simParam)) {

        if (!exists("SP", envir = .GlobalEnv)) {
            stop(
                paste(
                    "No SimParam object was supplied and no global object",
                    "named 'SP' was found."
                )
            )
        }

        simParam <- get(
            "SP",
            envir = .GlobalEnv
        )
    }

    marker_matrix <- AlphaSimR::pullSnpGeno(
        candidate_population,
        snpChip = snp_chip,
        simParam = simParam
    )

    marker_matrix <- as.matrix(
        marker_matrix
    )

    storage.mode(marker_matrix) <- "numeric"

    if (nrow(marker_matrix) != candidate_population@nInd) {
        stop(
            "The marker-matrix row count does not match population size."
        )
    }

    if (ncol(marker_matrix) < 1) {
        stop(
            "The marker matrix contains no SNP markers."
        )
    }

    if (anyNA(marker_matrix)) {
        stop(
            "The AlphaSimR marker matrix unexpectedly contains missing values."
        )
    }

    valid_values <- marker_matrix %in% c(
        0,
        1,
        2
    )

    if (!all(valid_values)) {
        stop(
            "The marker matrix contains values outside dosage coding {0,1,2}."
        )
    }

    ###########################################################################
    # rrBLUP::A.mat expects markers coded as:
    #
    #   -1 = first homozygote
    #    0 = heterozygote
    #    1 = second homozygote
    #
    # AlphaSimR provides:
    #
    #    0, 1, 2
    #
    # Therefore:
    #
    #   rrBLUP coding = AlphaSimR coding - 1
    ###########################################################################

    marker_matrix_rrblup <- marker_matrix - 1

    rownames(marker_matrix_rrblup) <- as.character(
        candidate_population@id
    )

    return(marker_matrix_rrblup)
}


###############################################################################
# 4. Calculate prediction uncertainty using G-BLUP
###############################################################################

compute_prediction_uncertainty <- function(
    candidate_population,
    training_population,
    trait = 1,
    snp_chip = 1,
    min_maf = NULL,
    max_missing = NULL,
    n_cores = 1,
    simParam = NULL
) {

    validate_uncertainty_inputs(
        candidate_population = candidate_population,
        training_population = training_population,
        trait = trait,
        snp_chip = snp_chip
    )


    ###########################################################################
    # Validate optional arguments
    ###########################################################################

    if (
        !is.numeric(n_cores) ||
        length(n_cores) != 1 ||
        n_cores %% 1 != 0 ||
        n_cores < 1
    ) {
        stop(
            "'n_cores' must be one positive whole number."
        )
    }

    if (
        !is.null(min_maf) &&
        (
            !is.numeric(min_maf) ||
            length(min_maf) != 1 ||
            min_maf < 0 ||
            min_maf > 0.5
        )
    ) {
        stop(
            "'min_maf' must be NULL or a number between 0 and 0.5."
        )
    }

    if (
        !is.null(max_missing) &&
        (
            !is.numeric(max_missing) ||
            length(max_missing) != 1 ||
            max_missing < 0 ||
            max_missing > 1
        )
    ) {
        stop(
            "'max_missing' must be NULL or a number between 0 and 1."
        )
    }


    ###########################################################################
    # Retrieve SimParam when omitted
    ###########################################################################

    if (is.null(simParam)) {

        if (!exists("SP", envir = .GlobalEnv)) {
            stop(
                paste(
                    "No SimParam object was supplied and no global object",
                    "named 'SP' was found."
                )
            )
        }

        simParam <- get(
            "SP",
            envir = .GlobalEnv
        )
    }


    ###########################################################################
    # Extract marker genotypes for every current candidate
    ###########################################################################

    marker_matrix <- prepare_candidate_markers(
        candidate_population = candidate_population,
        snp_chip = snp_chip,
        simParam = simParam
    )


    ###########################################################################
    # Build the genomic additive relationship matrix
    #
    # rrBLUP::A.mat expects markers coded {-1, 0, 1}.
    ###########################################################################

    genomic_relationship_matrix <- rrBLUP::A.mat(
        X = marker_matrix,
        min.MAF = min_maf,
        max.missing = max_missing,
        impute.method = "mean",
        n.core = as.integer(n_cores),
        shrink = FALSE
    )

    genomic_relationship_matrix <- as.matrix(
        genomic_relationship_matrix
    )

    candidate_ids <- as.character(
        candidate_population@id
    )

    rownames(genomic_relationship_matrix) <- candidate_ids
    colnames(genomic_relationship_matrix) <- candidate_ids


    ###########################################################################
    # Build the phenotyped training-data table
    ###########################################################################

    training_ids <- as.character(
        training_population@id
    )

    training_phenotypes <- as.numeric(
        AlphaSimR::pheno(
            training_population
        )[, trait]
    )

    training_data <- data.frame(
        individual_id = training_ids,
        phenotype = training_phenotypes,
        stringsAsFactors = FALSE
    )


    ###########################################################################
    # Fit G-BLUP and request formal prediction error variances
    #
    # kin.blup returns predictions and PEV for every candidate represented
    # in the relationship matrix, including candidates without phenotypes.
    ###########################################################################

    uncertainty_model <- rrBLUP::kin.blup(
        data = training_data,
        geno = "individual_id",
        pheno = "phenotype",
        K = genomic_relationship_matrix,
        GAUSS = FALSE,
        PEV = TRUE
    )


    ###########################################################################
    # Extract outputs and enforce candidate order
    ###########################################################################

    predicted_values <- as.numeric(
        uncertainty_model$g[
            candidate_ids
        ]
    )

    prediction_error_variance <- as.numeric(
        uncertainty_model$PEV[
            candidate_ids
        ]
    )

    if (
        length(predicted_values) !=
        candidate_population@nInd
    ) {
        stop(
            "The number of G-BLUP predictions does not match population size."
        )
    }

    if (
        length(prediction_error_variance) !=
        candidate_population@nInd
    ) {
        stop(
            "The number of PEV values does not match population size."
        )
    }

    if (anyNA(predicted_values)) {
        stop(
            "G-BLUP returned missing predicted values."
        )
    }

    if (anyNA(prediction_error_variance)) {
        stop(
            "G-BLUP returned missing PEV values."
        )
    }

    if (any(prediction_error_variance < -1e-8)) {
        stop(
            "G-BLUP returned materially negative PEV values."
        )
    }

    # Very small negative values can occur through floating-point rounding.
    prediction_error_variance <- pmax(
        prediction_error_variance,
        0
    )


    ###########################################################################
    # Calculate reliability
    #
    # reliability_i = 1 - PEV_i / (Vg * K_ii)
    ###########################################################################

    prior_genetic_variance <- (
        as.numeric(uncertainty_model$Vg) *
        diag(genomic_relationship_matrix)
    )

    reliability_raw <- 1 - (
        prediction_error_variance /
        prior_genetic_variance
    )

    # Keep the raw value for diagnostics and a bounded value for reporting.
    reliability <- pmin(
        pmax(reliability_raw, 0),
        1
    )


    ###########################################################################
    # Mark which candidates already have phenotypes
    ###########################################################################

    is_phenotyped <- candidate_ids %in% training_ids

    candidate_index <- seq_len(
        candidate_population@nInd
    )

    uncertainty_table <- data.frame(
        population_index = candidate_index,
        individual_id = candidate_ids,
        is_phenotyped = is_phenotyped,
        predicted_value = predicted_values,
        prediction_error_variance = prediction_error_variance,
        prediction_standard_error = sqrt(
            prediction_error_variance
        ),
        reliability_raw = reliability_raw,
        reliability = reliability,
        stringsAsFactors = FALSE
    )


    ###########################################################################
    # Create a table of only unphenotyped candidates, ranked by uncertainty
    ###########################################################################

    unphenotyped_uncertainty <- uncertainty_table[
        !uncertainty_table$is_phenotyped,
        ,
        drop = FALSE
    ]

    unphenotyped_uncertainty <- unphenotyped_uncertainty[
        order(
            unphenotyped_uncertainty$prediction_error_variance,
            decreasing = TRUE
        ),
        ,
        drop = FALSE
    ]

    unphenotyped_uncertainty$uncertainty_rank <- seq_len(
        nrow(unphenotyped_uncertainty)
    )

    rownames(uncertainty_table) <- NULL
    rownames(unphenotyped_uncertainty) <- NULL


    ###########################################################################
    # Return all useful results
    ###########################################################################

    return(
        list(
            model = uncertainty_model,
            genomic_relationship_matrix =
                genomic_relationship_matrix,
            uncertainty_table =
                uncertainty_table,
            unphenotyped_uncertainty =
                unphenotyped_uncertainty,
            training_population_size =
                training_population@nInd,
            candidate_population_size =
                candidate_population@nInd,
            genetic_variance =
                as.numeric(uncertainty_model$Vg),
            residual_variance =
                as.numeric(uncertainty_model$Ve)
        )
    )
}


###############################################################################
# 5. Select the most uncertain unphenotyped candidates
###############################################################################

select_highest_pev_candidates <- function(
    uncertainty_result,
    number_to_select
) {

    if (!is.list(uncertainty_result)) {
        stop(
            paste0(
            "'uncertainty_result' must be returned by ",
            "compute_prediction_uncertainty()."
    )
)
    }

    if (!"unphenotyped_uncertainty" %in% names(uncertainty_result)) {
        stop(
            paste0(
                "'uncertainty_result' does not contain ",
                "'unphenotyped_uncertainty'.")
        )
    }

    if (
        !is.numeric(number_to_select) ||
        length(number_to_select) != 1 ||
        number_to_select %% 1 != 0 ||
        number_to_select < 1
    ) {
        stop(
            "'number_to_select' must be one positive whole number."
        )
    }

    ranked_candidates <- uncertainty_result$unphenotyped_uncertainty

    if (number_to_select > nrow(ranked_candidates)) {
        stop(
            paste0(
                "'number_to_select' cannot exceed the number of ",
                "unphenotyped candidates, which is ",
                nrow(ranked_candidates),
                "."
            )
        )
    }

    selected_candidates <- ranked_candidates[
        seq_len(number_to_select),
        ,
        drop = FALSE
    ]

    return(selected_candidates)
}