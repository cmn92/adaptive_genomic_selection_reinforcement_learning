#########################################################################
# test_phenotyping.R
#
# Purpose:
# Test the phenotyping functions using the initial candidate population.
#########################################################################

library(AlphaSimR)

setwd("~/adaptive_genomic_selection_reinforcement_learning")

#########################################################################
# 1. Load the phenotyping functions
#########################################################################

source("src/simulator/phenotyping.R")


#########################################################################
# 2. Load the candidate population
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

cat("Objects loaded:\n")
print(loaded_objects)


#########################################################################
# 3. Check required objects
#########################################################################

if (!exists("candidate_population")) {
  stop("'candidate_population' was not found in the loaded file.")
}

if (!exists("SP")) {
  stop("'SP' was not found in the loaded file.")
}


#########################################################################
# 4. Randomly phenotype 200 of the 1,000 candidates
#########################################################################

phenotyping_result <- phenotype_random_sample(
  population = candidate_population,
  number_to_phenotype = 200,
  reps = 1,
  seed = 12345,
  simParam = SP
)


#########################################################################
# 5. Inspect results
#########################################################################

phenotyped_population <- phenotyping_result$phenotyped_population
phenotype_table <- phenotyping_result$phenotype_table

cat("\nPhenotyping completed successfully.\n")
cat(
  "Candidate population size:",
  candidate_population@nInd,
  "\n"
)

cat(
  "Number phenotyped:",
  phenotyped_population@nInd,
  "\n"
)

cat("\nFirst six phenotype records:\n")
print(head(phenotype_table))


#########################################################################
# 6. Basic checks
#########################################################################

stopifnot(
  phenotyped_population@nInd == 200
)

stopifnot(
  nrow(phenotype_table) == 200
)

stopifnot(
  length(unique(phenotype_table$individual_id)) == 200
)

stopifnot(
  all(!is.na(phenotype_table$phenotype_trait_1))
)

cat("\nAll phenotyping checks passed.\n")