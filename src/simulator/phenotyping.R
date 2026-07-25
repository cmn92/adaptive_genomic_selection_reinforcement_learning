###############################################################################
# phenotyping.R
#
# Purpose:
# Provide functions for phenotyping a selected subset of candidate lines.
#
# A phenotyping strategy supplies the IDs or row indices of individuals to
# phenotype and this module then generates their observed phenotypes using
# AlphaSimR
#################################################################################

###################################################################################
# 1. Load required packages
###################################################################################

library(AlphaSimR)

###############################################################################
# 2. Validate selected individual indices
###############################################################################
validate_selected_indices <- function(population, selected_indices) {
    if (!inherits(population, "Pop")) {
        stop("The provided population is not a valid AlphaSimR Pop object.")
    }

    if (length(selected_indices) == 0) {
        stop("The selected_indices vector is empty. Please provide valid indices.")
    }

    if (!is.numeric(selected_indices) || any(selected_indices < 1) || any(selected_indices > nInd(population))) {
        stop("Selected indices are out of bounds. They must be numeric and within the range of the population size.")
    }

    if (any(is.na(selected_indices))) {
        stop("Selected indices contain NA values. Please provide valid indices.")
    }

    if (any(selected_indices %% 1 != 0)) {
        stop("Selected indices must be whole numbers (integers).")
    }

    if (any(selected_indices  > population@nInd)) {
        stop(
            paste0(
                "At least one selected index exceeds the number of individuals in the population (",
                population@nInd,
                ")."
            )
        )
    }

    if (any(duplicated(selected_indices))) {
        stop("Selected indices contain duplicates. Please provide unique indices.")
    }

    return(as.integer(selected_indices) )

}

################################################################################
# 3. Phenotype selected individuals
################################################################################

phenotype_selected <- function(
    population,
    selected_indices,
    reps = 1,
    simParam = NULL
) {
    # Validate the selected indices
    selected_indices <- validate_selected_indices(population, selected_indices)

    if (!is.numeric(reps) || length(reps) != 1 || reps < 1 || reps %% 1 != 0) {
        stop("The 'reps' parameter must be a single positive integer.")
    }

    if (is.null(simParam)) {
        if(!exists("SP", envir =.GlobalEnv)) {
            stop(
                paste(
                    "No SimParam object was supplied and no global object named",
                    "'SP' was found."
                )
            )
        }

        simParam <- get("SP", envir = .GlobalEnv)
    }

################################################################################
# Extract only the selected individuals for phenotyping
################################################################################
    phenotype_population <- population[selected_indices]


################################################################################
# Generate observed phenotypes
#
# Environmental variance was previously specified using SP$setVarE().
# thus setPheno() uses the error variance stored in SP
##############################################################################
    phenotyped_population <- setPheno(
        pop=phenotype_population,
        reps = as.integer(reps),
        simParam = simParam
    )

################################################################################
# Create a clean phenotype table
################################################################################
    phenotype_matrix <- phenotyped_population@pheno
    genetic_value_matrix <- phenotyped_population@gv

    phenotype_table <- data.frame(
        population_index = selected_indices,
        individual_id = phenotyped_population@id,
        stringsAsFactors = FALSE
    )

    for (trait_index in seq_len(ncol(phenotype_matrix))) {

        phenotype_table[[paste0("phenotype_trait_", trait_index)]] <- phenotype_matrix[, trait_index]

        phenotype_table[[paste0("true_genetic_value_trait_", trait_index)]] <- genetic_value_matrix[, trait_index]
  }

    phenotype_table$replications <- as.integer(reps)

################################################################################
# Return all useful objects
################################################################################
    return(list(
        phenotyped_population = phenotyped_population,
        phenotype_table = phenotype_table,
        selected_indices = selected_indices
    ))
}


#########################################################################
# 4. Randomly select and phenotype candidates
#
# This function is mainly for testing. Later, random sampling will live
# in src/baselines/random_sampling.py or an equivalent R baseline module.
#########################################################################

phenotype_random_sample <- function(
    population,
    number_to_phenotype,
    reps = 1,
    seed = NULL,
    simParam = NULL
) {

  if (
    !is.numeric(number_to_phenotype) ||
    length(number_to_phenotype) != 1 ||
    number_to_phenotype %% 1 != 0
  ) {
    stop("'number_to_phenotype' must be one whole number.")
  }

  if (number_to_phenotype < 1) {
    stop("'number_to_phenotype' must be at least 1.")
  }

  if (number_to_phenotype > population@nInd) {
    stop(
      paste0(
        "'number_to_phenotype' cannot exceed population size ",
        population@nInd,
        "."
      )
    )
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  selected_indices <- sample(
    x = seq_len(population@nInd),
    size = number_to_phenotype,
    replace = FALSE
  )

  phenotype_selected(
    population = population,
    selected_indices = selected_indices,
    reps = reps,
    simParam = simParam
  )
}